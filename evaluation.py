from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
from shared.utils import load_from_json
import textdistance
import pandas as pd
import numpy as np
import os.path

pd.set_option('display.max_rows', 100)

def jaccard_similarity(doc1, doc2): 
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

def levenstein_distance(doc1, doc2):
    lv_dist = textdistance.levenshtein.normalized_similarity(doc1, doc2)
    return lv_dist

def get_relevance_label_df(query_answer_pair_filepath):
    query_answer_pair = load_from_json(query_answer_pair_filepath)
    relevance_label_df = pd.DataFrame.from_records(query_answer_pair)
    return relevance_label_df

def get_relevance_label(relevance_label_df):
    relevance_label_df.rename(columns={'question': 'query_string'}, inplace=True)
    relevance_label = relevance_label_df.groupby(['query_string'])['answer'].apply(list).to_dict()
    return relevance_label

class Result:
    """ Class for saving evaluation metrics results in a dictionary data structure 
    
    :param method: Supervised or Unsupervised
    :param match_field: answer / question / question_answer / question_answer_concat
    :param rank_field: BERT-Q-a / BERT-Q-q
    :param loss_type: triplet / softmax
    :param query_type: training data as faq or user_query
    :param neg_type: simple or hard
    :param ndcg3: evaluation metric score NDCG@3
    :param ndcg5: evaluation metric score NDCG@5
    :param ndcg20: evaluation metric score NDCG@10
    :param p3: evaluation metric score P@3
    :param p5: evaluation metric score P@5
    :param p20: evaluation metric score P@10
    :param _map: evaluation metric score MAP
    """
    def __init__(self, method="", match_field="", rank_field="", loss_type="", query_type="", neg_type="", 
                 ndcg2=0, ndcg3=0, ndcg5=0, p2=0, p3=0, p5=0, _map=0):
        
        self.method = method
        self.match_field = match_field
        self.rank_field = rank_field
        self.loss_type = loss_type
        self.query_type = query_type
        self.neg_type = neg_type
        self.ndcg2 = ndcg2
        self.ndcg3  = ndcg3
        self.ndcg5  = ndcg5
        self.p2    = p2
        self.p3     = p3
        self.p5     = p5
        self._map   =  _map
        
    def __repr__(self):
        return {
            "Method"            : self.method.capitalize(),
            "Matching Field"    : self.match_field,
            "Ranking Field"     : self.rank_field,
            "Loss"              : self.loss_type,
            "Training Data"     : self.query_type,
            "Negative Sampling" : self.neg_type,
            "NDCG@2"            : "{0:.4f}".format(self.ndcg2),
            "NDCG@3"            : "{0:.4f}".format(self.ndcg3),
            "NDCG@5"            : "{0:.4f}".format(self.ndcg5),
            "P@2"               : "{0:.4f}".format(self.p2),
            "P@3"               : "{0:.4f}".format(self.p3),
            "P@5"               : "{0:.4f}".format(self.p5),
            "MAP"               : "{0:.4f}".format(self._map)
        }

class Evaluation(object):
    """ Class for generating evaluation of re-ranked results 
    
    :param qas_filename: evaluating test queries using query_answer_pairs.json / synthetic_query_answer_pairs.json file
    
    :param rank_results_filepath: filepath to rank results
        BERT-FAQ/data/CovidFAQ/rank_results         # CovidFAQ
        BERT-FAQ/data/StackFAQ/rank_results         # StackFAQ
        BERT-FAQ/data/FAQIR/rank_results            # FAQIR
    :param jc_threshold: jaccard similarity threshold
    :param test_data: param used for generating evaluation for synthetic/user_query test data
    :param rankers: rankers e.g. unsupervised, supervised
    :param rank_fields: rank fields e.g. BERT-Q-a, BERT-Q-q
    :param loss_types:  loss types e.g. triplet, softmax
    :param query_types: query types e.g. faq, user_query
    :param neg_types: negative types e.g. simple, hard
    :param top_k: top k e.g. 2, 3, 5
    """

    def __init__(self, qas_filename, rank_results_filepath, jc_threshold=1.0, test_data="synthetic", rankers=["unsupervised", "supervised"], 
                 rank_fields=["BERT-Q-a", "BERT-Q-q"], loss_types=["triplet", "softmax"], query_types=["faq", "user_query"], 
                 neg_types=["simple", "hard"], top_k=[2, 3, 5]):
        
        if test_data not in {'synthetic', 'user_query'}:
            raise ValueError('error, test_data not exist')

        self.top_k = top_k
        self.rankers = rankers
        self.neg_types = neg_types
        self.test_data = test_data
        self.loss_types = loss_types
        self.rank_fields = rank_fields
        self.query_types = query_types
        self.rank_results_filepath = rank_results_filepath

        self.ndcg_per_query = []
        self.prec_per_query = []
        self.map_per_query = []

        list_of_qas = load_from_json(qas_filename)

        total_questions = 0
        filtered_questions = 0

        self.valid_queries = []
        
        for item in list_of_qas:
            total_questions += 1
            if 'jc_sim' in item:
                jc = float(item['jc_sim'])
                if jc <= jc_threshold:
                    filtered_questions += 1
                    self.valid_queries.append(item['question'])
    
    def compute_map(self, result_filepath, ranker, match_field, rank_field="", loss_type="", query_type="", neg_type=""):
        """ Compute average precision score for a set of rank results
        
        :param result_filepath:  filepath to Elasticsearch rank results
        :param ranker: supervised / unsupervised
        :param match_field: answer / question / question_answer / question_answer_concat
        :param loss_type: triplet / softmax
        :param rank_field: BERT-Q-a / BERT-Q-q
        :param query_type: faq / user_query
        :param neg_type: simple / hard
        """
        query_results = load_from_json(result_filepath)

        sum_ap = 0
        num_queries = 0
        map_per_query = []
        for result in query_results:
            query_string = result['query_string']

            if query_string in self.valid_queries:

                topk_results = result['rerank_preds']

                labels = []
                reranks = []

                for topk in topk_results:
                    labels.append(topk['label'])
                    reranks.append(topk['score'])

                true_relevance = np.array(labels)
                scores = np.array(reranks)

                ap = 0
                all_zeros = not np.any(labels)
                if labels and reranks and not all_zeros:
                    ap = average_precision_score(true_relevance, scores)

                sum_ap = sum_ap + ap
                num_queries = num_queries + 1

                query_map = {
                    "Query": query_string,
                    "MAP": ap,
                    "Method": ranker,
                    "Matching Field": match_field,
                    "Ranking Field": rank_field,
                    "Loss": loss_type,
                    "Training Data": query_type,
                    "Negative Sampling": neg_type
                }
                map_per_query.append(query_map)

        return (float(sum_ap / num_queries)), map_per_query

    def compute_prec(self, result_filepath, k, ranker, match_field, rank_field="", loss_type="", query_type="", neg_type=""):
        """ Compute precision score for a set of rank results
        
        :param result_filepath:  filepath to Elasticsearch rank results
        :param ranker: supervised / unsupervised
        :param match_field: answer / question / question_answer / question_answer_concat
        :param loss_type: triplet / softmax
        :param rank_field: BERT-Q-a / BERT-Q-q
        :param query_type: faq / user_query
        :param neg_type: simple / hard
        """
        query_results = load_from_json(result_filepath)

        sum_prec = 0
        num_queries = 0
        prec_per_query = []
        for result in query_results:
            query_string = result['query_string']

            if query_string in self.valid_queries:
                topk_results = result['rerank_preds']

                labels = []
                reranks = []

                for topk in topk_results[:k]:
                    labels.append(topk['label'])
                    reranks.append(topk['score'])

                true_relevance = np.array(labels)
                scores = np.array(reranks)

                prec = 0
                all_zeros = not np.any(labels)
                if labels and reranks and not all_zeros:
                    prec = sum(true_relevance) / len(true_relevance)

                sum_prec = sum_prec + prec
                num_queries = num_queries + 1

                query_prec = {
                    "Query": query_string,
                    "k": k,
                    "Prec": prec,
                    "Method": ranker,
                    "Matching Field": match_field,
                    "Ranking Field": rank_field,
                    "Loss": loss_type,
                    "Training Data": query_type,
                    "Negative Sampling": neg_type
                }
                prec_per_query.append(query_prec)

        return (float(sum_prec / num_queries)), prec_per_query

    def compute_ndcg(self, result_filepath, k, ranker, match_field, rank_field="", loss_type="", query_type="", neg_type=""):
        """ Compute NDCG score for a set of rank results
        
        :param result_filepath:  filepath to Elasticsearch rank results
        :param ranker: supervised / unsupervised
        :param match_field: answer / question / question_answer / question_answer_concat
        :param loss_type: triplet / softmax
        :param rank_field: BERT-Q-a / BERT-Q-q
        :param query_type: faq / user_query
        :param neg_type: simple / hard
        """
        query_results = load_from_json(result_filepath)

        sum_ndcg = 0
        num_queries = 0
        ndcg_per_query = []

        for result in query_results:
            query_string = result['query_string']
         
            if query_string in self.valid_queries:
                
                topk_results = result['rerank_preds']

                labels = []
                reranks = []

                for topk in topk_results[:k]:
                    labels.append(topk['label'])
                    reranks.append(topk['score'])

                true_relevance = np.asarray([labels])
                scores = np.asarray([reranks])
                
                ndcg = 0
                if labels and reranks:
                    ndcg = ndcg_score(true_relevance, scores)

                sum_ndcg = sum_ndcg + ndcg
                num_queries = num_queries + 1

                query_ndcg = {
                    "Query": query_string,
                    "k": k,
                    "NDCG": ndcg,
                    "Method": ranker,
                    "Matching Field": match_field,
                    "Ranking Field": rank_field,
                    "Loss": loss_type,
                    "Training Data": query_type,
                    "Negative Sampling": neg_type
                }
                ndcg_per_query.append(query_ndcg)

        return (float(sum_ndcg / num_queries)), ndcg_per_query
    
    def get_eval_output(self):
        """ Generate evaluation metrics and save them into a dictionary

        :return: Python dictionary
        """
        output = dict()
        output['eval'] = dict()

        for ranker in self.rankers:
    
            # Compute metrics for the unsupervised method
            if ranker == "unsupervised":
                
                file_path = ""
                if self.test_data == 'synthetic':
                    file_path = self.rank_results_filepath + "/" + ranker + "/synthetic"
                elif self.test_data == 'user_query':
                    file_path = self.rank_results_filepath + "/" + ranker + "/user_query"
                
                method = ranker
                answer_metric = []
                question_metric = []
                question_answer_metric = []
                question_answer_concat_metric = []

                for k in self.top_k:
                    # compute NDCG@k
                    
                    avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/es_query_by_answer.json", k, ranker, "answer")
                    answer_metric.append(avg_ndcg)
                    self.ndcg_per_query += ndcg_per_query

                    avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/es_query_by_question.json", k, ranker, "question")
                    question_metric.append(avg_ndcg)
                    self.ndcg_per_query += ndcg_per_query
                    
                    avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/es_query_by_question_answer.json", k, ranker, "question_answer")
                    question_answer_metric.append(avg_ndcg)
                    self.ndcg_per_query += ndcg_per_query

                    avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/es_query_by_question_answer_concat.json", k, ranker, "question_answer_concat")
                    question_answer_concat_metric.append(avg_ndcg)
                    self.ndcg_per_query += ndcg_per_query

                    # compute P@k
                    avg_prec, prec_per_query = self.compute_prec(file_path + "/es_query_by_answer.json", k, ranker, "answer")
                    answer_metric.append(avg_prec)
                    self.prec_per_query += prec_per_query

                    avg_prec, prec_per_query = self.compute_prec(file_path + "/es_query_by_question.json", k, ranker, "question")
                    question_metric.append(avg_prec)
                    self.prec_per_query += prec_per_query

                    avg_prec, prec_per_query = self.compute_prec(file_path + "/es_query_by_question_answer.json", k, ranker, "question_answer")
                    question_answer_metric.append(avg_prec)
                    self.prec_per_query += prec_per_query

                    avg_prec, prec_per_query = self.compute_prec(file_path + "/es_query_by_question_answer_concat.json", k, ranker, "question_answer_concat")
                    question_answer_concat_metric.append(avg_prec)
                    self.prec_per_query += prec_per_query

                # compute MAP
                avg_map, map_per_query = self.compute_map(file_path + "/es_query_by_answer.json", ranker, "answer")
                answer_metric.append(avg_map)
                self.map_per_query += map_per_query

                avg_map, map_per_query = self.compute_map(file_path + "/es_query_by_question.json", ranker, "question")
                question_metric.append(avg_map)
                self.map_per_query += map_per_query

                avg_map, map_per_query = self.compute_map(file_path + "/es_query_by_question_answer.json", ranker, "question_answer")
                question_answer_metric.append(avg_map)
                self.map_per_query += map_per_query

                avg_map, map_per_query = self.compute_map(file_path + "/es_query_by_question_answer_concat.json", ranker, "question_answer_concat")
                question_answer_concat_metric.append(avg_map)
                self.map_per_query += map_per_query

                result = Result(method="unsupervised", match_field="answer", 
                    ndcg2=answer_metric[0], ndcg3=answer_metric[1], ndcg5=answer_metric[2], 
                    p2=answer_metric[3], p3=answer_metric[4], p5=answer_metric[5], _map=answer_metric[6]
                )
                
                output['eval'][method + "_answer"] = result.__repr__()

                result = Result(method="unsupervised", match_field="question",
                    ndcg2=question_metric[0], ndcg3=question_metric[1], ndcg5=question_metric[2], 
                    p2=question_metric[3], p3=question_metric[4], p5=question_metric[5], _map=question_metric[6]
                )
                
                output['eval'][method + "_question"] = result.__repr__()
                
                result = Result(method="unsupervised", match_field="question_answer",
                    ndcg2=question_answer_metric[0], ndcg3=question_answer_metric[1], ndcg5=question_answer_metric[2], 
                    p2=question_answer_metric[3], p3=question_answer_metric[4], p5=question_answer_metric[5], _map=question_answer_metric[6]
                )
                
                output['eval'][method + "_question_answer"] = result.__repr__()
                            
                result = Result(method="unsupervised", match_field="question_answer_concat",
                    ndcg2=question_answer_concat_metric[0], ndcg3=question_answer_concat_metric[1], ndcg5=question_answer_concat_metric[2], 
                    p2=question_answer_concat_metric[3], p3=question_answer_concat_metric[4], p5=question_answer_concat_metric[5], 
                    _map=question_answer_concat_metric[6]
                )
                
                output['eval'][method + "_question_answer_concat"] = result.__repr__()
                
            elif ranker == 'supervised':
            
                # Compute metrics for the supervised method
                
                for rank_field in self.rank_fields:
                
                    for loss_type in self.loss_types:

                        for query_type in self.query_types:

                            for neg_type in self.neg_types:

                                file_path = ""
                                if self.test_data == 'synthetic':
                                    file_path = self.rank_results_filepath + "/" + ranker + "/synthetic/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type
                                elif self.test_data == 'user_query':
                                    file_path = self.rank_results_filepath + "/" + ranker + "/user_query/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type

                                if os.path.isdir(file_path):

                                    method = loss_type + "_" + neg_type + "_" + query_type + "_" + rank_field

                                    answer_metric = []
                                    question_metric = []
                                    question_answer_metric = []
                                    question_answer_concat_metric = []

                                    for k in self.top_k:

                                        # compute NDCG@k
                                        avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/reranked_query_by_answer.json", k, ranker, "answer", rank_field, loss_type, query_type, neg_type)
                                        answer_metric.append(avg_ndcg)
                                        self.ndcg_per_query += ndcg_per_query

                                        avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/reranked_query_by_question.json", k, ranker, "question", rank_field, loss_type, query_type, neg_type)
                                        question_metric.append(avg_ndcg)
                                        self.ndcg_per_query += ndcg_per_query

                                        avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/reranked_query_by_question_answer.json", k, ranker, "question_answer", rank_field, loss_type, query_type, neg_type)
                                        question_answer_metric.append(avg_ndcg)
                                        self.ndcg_per_query += ndcg_per_query

                                        avg_ndcg, ndcg_per_query = self.compute_ndcg(file_path + "/reranked_query_by_question_answer_concat.json", k, ranker, "question_answer_concat", rank_field, loss_type, query_type, neg_type)
                                        question_answer_concat_metric.append(avg_ndcg)
                                        self.ndcg_per_query += ndcg_per_query


                                        # compute P@k
                                        avg_prec, prec_per_query = self.compute_prec(file_path + "/reranked_query_by_answer.json", k, ranker, "answer", rank_field, loss_type, query_type, neg_type)
                                        answer_metric.append(avg_prec)
                                        self.prec_per_query += prec_per_query

                                        avg_prec, prec_per_query = self.compute_prec(file_path + "/reranked_query_by_question.json", k, ranker, "question", rank_field, loss_type, query_type, neg_type)
                                        question_metric.append(avg_prec)
                                        self.prec_per_query += prec_per_query

                                        avg_prec, prec_per_query = self.compute_prec(file_path + "/reranked_query_by_question_answer.json", k, ranker, "question_answer", rank_field, loss_type, query_type, neg_type)
                                        question_answer_metric.append(avg_prec)
                                        self.prec_per_query += prec_per_query

                                        avg_prec, prec_per_query = self.compute_prec(file_path + "/reranked_query_by_question_answer_concat.json", k, ranker, "question_answer_concat", rank_field, loss_type, query_type, neg_type)
                                        question_answer_concat_metric.append(avg_prec)
                                        self.prec_per_query += prec_per_query

                                    # compute map
                                    avg_map, map_per_query = self.compute_map(file_path + "/reranked_query_by_answer.json", ranker, "answer", rank_field, loss_type, query_type, neg_type)
                                    answer_metric.append(avg_map)
                                    self.map_per_query += map_per_query

                                    avg_map, map_per_query = self.compute_map(file_path + "/reranked_query_by_question.json", ranker, "question", rank_field, loss_type, query_type, neg_type)
                                    question_metric.append(avg_map)
                                    self.map_per_query += map_per_query

                                    avg_map, map_per_query = self.compute_map(file_path + "/reranked_query_by_question_answer.json", ranker, "question_answer", rank_field, loss_type, query_type, neg_type)
                                    question_answer_metric.append(avg_map)
                                    self.map_per_query += map_per_query

                                    avg_map, map_per_query = self.compute_map(file_path + "/reranked_query_by_question_answer_concat.json", ranker, "question_answer_concat", rank_field, loss_type, query_type, neg_type)
                                    question_answer_concat_metric.append(avg_map)
                                    self.map_per_query += map_per_query

                                    result = Result(
                                        method="supervised", match_field="answer", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg2=answer_metric[0], ndcg3=answer_metric[1], ndcg5=answer_metric[2], 
                                        p2=answer_metric[3], p3=answer_metric[4], p5=answer_metric[5], _map=answer_metric[6]
                                    )

                                    output['eval'][method + "_answer"] = result.__repr__()
    
                                    result = Result(
                                        method="supervised", match_field="question", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg2=question_metric[0], ndcg3=question_metric[1], ndcg5=question_metric[2], 
                                        p2=question_metric[3], p3=question_metric[4], p5=question_metric[5], _map=question_metric[6]
                                    )

                                    output['eval'][method + "_question"] = result.__repr__()

                                    result = Result(
                                        method="supervised", match_field="question_answer", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg2=question_answer_metric[0], ndcg3=question_answer_metric[1], ndcg5=question_answer_metric[2], 
                                        p2=question_answer_metric[3], p3=question_answer_metric[4], p5=question_answer_metric[5], _map=question_answer_metric[6]
                                    )

                                    output['eval'][method + "_question_answer"] = result.__repr__()
                                    
                                    result = Result(
                                        method="supervised", match_field="question_answer_concat", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg2=question_answer_concat_metric[0], ndcg3=question_answer_concat_metric[1], ndcg5=question_answer_concat_metric[2], 
                                        p2=question_answer_concat_metric[3], p3=question_answer_concat_metric[4], p5=question_answer_concat_metric[5], _map=question_answer_concat_metric[6]
                                    )
                                    
                                    output['eval'][method + "_question_answer_concat"] = result.__repr__()      
        
        return output

    def get_eval_df(self):
        """ Generate evaluation DataFrame from rank_results filepath 
        
        :return: evaluation DataFrame
        """
        output = self.get_eval_output()
        df = pd.DataFrame.from_dict({(i,j): output[i][j] 
                                                for i in output.keys() 
                                                    for j in output[i].keys()},
                                                                orient='index')

        df.reset_index(drop=True, inplace=True)
        return df
    
