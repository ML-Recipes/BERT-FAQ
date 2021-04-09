from faq_bert import FAQ_BERT
from searcher import Searcher

class FAQ_BERT_Ranker(object):
    """ Class to generate top-k ranked results for a given input query string 
    
    :param es: Elasticsearch instance
    :param index: Elasticsearch index name
    :param top_k: parameter used during model training (e.g top_k=100)
    :param bert_model_path: bert model path
    :param rank_field: BERT prediction for rank_field answer or question
    :param w_t: weight parameter used for re-ranking of ES score
    """
    def __init__(self, es, index, fields, top_k, bert_model_path, rank_field='BERT-Q-a', w_t=10):
        self.es = es
        self.index = index
        self.fields = fields
        self.top_k = top_k
        self.bert_model_path = bert_model_path
        self.rank_field = rank_field
        self.w_t = w_t
        
        self.searcher = Searcher(es, index, fields, top_k)

        self.es_topk_results = []
        self.bert_topk_preds = []
        self.ranked_results  = []

    def get_es_topk_results(self, query_string):
        """ Get Elasticsearch top-k results 
        
        :param query_string: Elasticsearch query string
        :return: ES top-k results
        """
        results = self.searcher.query(query_string)
           
        topk_results = []
        for doc in results:
            topk_results.append(
                {
                    "es_score": float("{0:.4f}".format(doc['score'])),
                    "question": doc['question'],
                    "answer": doc['answer']
                }
            )

        es_topk_results = dict()
        es_topk_results['query_string'] = query_string
        es_topk_results['topk_results'] = topk_results
    
        return es_topk_results

    def get_bert_topk_preds(self, es_topk_results):
        """ Get BERT top-k predictions by rank field 
        
        :param es_topk_results: Python dictionary
        :return: BERT predictions on ES top-k results
        """
        faq_bert = FAQ_BERT(bert_model_path=self.bert_model_path)

        bert_topk_preds = []
        
        query_string = es_topk_results['query_string']
        topk_results = es_topk_results['topk_results']

        for doc in topk_results:
            question = doc['question']
            answer = doc['answer']
            es_score = doc['es_score']
            
            bert_score = 0
            if self.rank_field == "BERT-Q-a":
                bert_score = faq_bert.predict(query_string, answer)
            elif self.rank_field == "BERT-Q-q":
                bert_score = faq_bert.predict(query_string, question)
            else:
                raise ValueError("error, no rank_field found for {}".format(self.rank_field))

            bert_topk_preds.append(
                {
                    "question": query_string,
                    "answer": answer,
                    "es_score": es_score,
                    "bert_score": float("{0:.4f}".format(bert_score))
                }
            )
        
        return bert_topk_preds

    def get_ranked_results(self, bert_topk_preds):
        """ Get top-k re-ranked results 
        
        :param bert_topk_preds: bert top-k results
        :return: ranked list of top-k results in descending order by score
        """
        norm_results = []
        for doc in bert_topk_preds:
            question = doc['question']
            answer = doc['answer']
            es_score = doc['es_score']
            bert_score = doc['bert_score']
            score = (self.w_t * es_score) + bert_score

            norm_results.append(
                {
                    "question": question,
                    "answer": answer,
                    "es_score": es_score,
                    "bert_score": bert_score,
                    "score": float("{0:.4f}".format(score))
                }
            )

        ranked_results = sorted(norm_results, key=lambda x: x['score'], reverse=True)
        
        return ranked_results

    def rank_results(self, query_string):
        """ Rank ES top-k results for a given input query string 
            using BERT pretrained model
        
        :param query_string: input query
        :return: ES top-k ranked results
        """
        es_topk_results = self.get_es_topk_results(query_string)
        bert_topk_preds = self.get_bert_topk_preds(es_topk_results)
        ranked_results = self.get_ranked_results(bert_topk_preds)

        self.es_topk_results = es_topk_results
        self.bert_topk_preds = bert_topk_preds
        self.ranked_results  = ranked_results
        
        return ranked_results


