

from shared.utils import load_from_json
from shared.utils import dump_to_json
from shared.utils import make_dirs
from evaluation import get_relevance_label_df
from evaluation import get_relevance_label
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import logging
import os

from searcher import Searcher
from faq_bert import FAQ_BERT

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

class ReRanker(object):
    """ Class for generating Elasticsearch, BERT, and top-k re-ranked results 

    :param bert_model_path: path residing finetuned BERT model
    :param test_queries: test queries used as query to Elasticsearch index
    :param relevance_label_df: dataframe of relevance labels
    :param loss_type: BERT model trained with loss type (triplet or softmax)
    """

    def __init__(self, bert_model_path, test_queries, relevance_label_df, loss_type="triplet"):
        
        self.bert_model_path = bert_model_path
        self.test_queries = test_queries
        self.loss_type = loss_type
        self.es_topk_results = []
        self.bert_topk_results = []
        self.reranked_results = []

        # Generate a dictionary where {key: query_string, value: list of answers}
        self.relevance_label = get_relevance_label(relevance_label_df)
        

    def get_es_topk_results(self, es, index, query_by, top_k):
        """
        Get top-k results from Elasticsearch querying index field(s)
        for each query string in valid_queries

        :param es: Elasticsearch instance
        :param index: Elasticsearch index
        :param query_by: Elasticsearch field(s) index
        :param top_k: Elasticsearch top-k results
        :return: list of query strings and associated ES top-k results
        """

        logging.info("Generating ES top-k results ...")
        
        # define Searcher class and query by fields
        s = Searcher(es, index=index, fields=query_by, top_k=top_k)

        es_topk_results = []
        for query_string in tqdm(self.test_queries):

            # perform querying on ES
            topk_results = s.query(query_string)

            # get the list of actual answers
            answers = self.relevance_label[query_string]
            
            # obtain relevance label for each answer
            topk_with_label = []
            for doc in topk_results:
                topk_answer = doc['answer']

                # check if the answer is a true answer
                label = 0
                if topk_answer in answers:
                    label = 1

                data = {
                    "score": doc['score'],
                    "question": query_string,
                    "answer": topk_answer,
                    "label": label
                }
                topk_with_label.append(data)

            es_topk_results.append({"query_string": query_string, "rerank_preds": topk_with_label})


        return es_topk_results

    def get_bert_topk_preds(self, all_results):
        """ 
        Predict similarity / label score for each question-answer pair
        
        :param all_results: Elasticsearch results
        :return: topk prediction list
        """
        
        logging.info("Generating BERT top-k results ...")
        
        faq_bert = FAQ_BERT(
            model_path=self.bert_model_path, loss_type=self.loss_type
        )

        bert_topk_results = []
        for result in tqdm(all_results):
            query_string = result['query_string']                          
            topk_results = result['rerank_preds']                        
            
            # get the list of actual answers
            answers = self.relevance_label[query_string]
            
            response = dict()
            topk_preds = []
            for elem in topk_results:
                es_score = elem['score']
                question = elem['question']
                answer = elem['answer']
                
                bert_score = faq_bert.predict(query_string, answer)
                
                # check if the answer is a true answer
                label = 0
                if answer in answers:
                    label = 1

                data = {
                    "es_score": es_score,
                    "question": question,
                    "answer": answer,
                    "bert_score": bert_score,
                    "label": label
                }
                topk_preds.append(data)
            
            response["query_string"] = query_string
            response["topk_preds"] = topk_preds
            bert_topk_results.append(response)
        
        return bert_topk_results

    def get_reranked_results(self, query_topk_preds):
        """
        Rank the top-k results for each query in query_topk_preds.
        We sum bert_score with query_score and sort the list in descending order by final score
    
        :param query_topk_preds: list consisting of query and topk prediction results
        :return: query_string and ranked top-k results list
        """
        
        logging.info("Re-ranking the top-k results ...")
        
        results = []
        for query_topk in query_topk_preds:
            query_string = query_topk['query_string']
            topk_preds = query_topk['topk_preds']
            
            norm_results = []
            for pred in topk_preds:
                question = pred['question']
                answer = pred['answer']
                score = pred['es_score'] + pred['bert_score']
                label = pred['label']

                result = {
                    'question': question,
                    'answer': answer, 
                    'score': score,
                    'label': label
                }
                norm_results.append(result)
            results.append({'query_string': query_string, 'norm_results': norm_results})

        # rank all topk predictions by final_score in descending order
        reranked_results = []
        for r in results:
            query_string = r['query_string']
            norm_results = r['norm_results']
            rerank_preds = sorted(norm_results, key=lambda x: x['score'], reverse=True)
            reranked_results.append({'query_string': query_string, 'rerank_preds': rerank_preds})
        return reranked_results

    def rank_results(self, es, index, query_by, top_k=10):
        """ Rank query results in Elasticsearch index 
        
        :param index: Elasticsearch instance
        :param index: Elasticsearch index
        :param query_by: Elasticsearch query field 
        :param top_k: top-k results
        """
        
        es_topk_results = self.get_es_topk_results(es=es, index=index, query_by=query_by, top_k=top_k)
        bert_topk_results = self.get_bert_topk_preds(es_topk_results)
        reranked_results = self.get_reranked_results(bert_topk_results)

        self.es_topk_results = es_topk_results
        self.bert_topk_results = bert_topk_results
        self.reranked_results = reranked_results
    
