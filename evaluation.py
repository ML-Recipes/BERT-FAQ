from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import ndcg_score
from shared.utils import load_from_json
from metric import NDCG
from tqdm import tqdm
import pandas as pd
import numpy as np
import os.path
import math

pd.set_option('display.max_rows', 100)

def get_relevance_label_df(query_answer_pair_filepath):
    query_answer_pair = load_from_json(query_answer_pair_filepath)
    relevance_label_df = pd.DataFrame.from_records(query_answer_pair)
    return relevance_label_df

def get_relevance_label(relevance_label_df):
    relevance_label_df.rename(columns={'question': 'query_string'}, inplace=True)
    relevance_label = relevance_label_df.groupby(['query_string'])['answer'].apply(list).to_dict()
    return relevance_label

def compute_map(result_filepath):
    query_results = load_from_json(result_filepath)

    sum_ap = 0
    num_queries = 0
    for result in query_results:
        query_string = result['query_string']
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

    return float(sum_ap / num_queries)

def compute_prec(result_filepath, k):
    query_results = load_from_json(result_filepath)

    sum_prec = 0
    num_queries = 0
    for result in query_results:
        query_string = result['query_string']
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

    return float(sum_prec / num_queries)

def compute_ndcg(result_filepath, k):
    query_results = load_from_json(result_filepath)

    sum_ndcg = 0
    num_queries = 0
    for result in query_results:
        query_string = result['query_string']
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

    return float(sum_ndcg / num_queries)

class Result:
    """ Class for saving evaluation metrics results in a dictionary data structure 
    
    :param method: Supervised or Unsupervised
    :param match_field: answer / question / question_answer / question_answer_concat
    :param rank_field: BERT-Q-a / BERT-Q-q
    :param loss_type: triplet or softmax
    :param query_type: training data as faq or user_query
    :param neg_type: simple or hard
    :param ndcg3: evaluation metric score NDCG@3
    :param ndcg5: evaluation metric score NDCG@5
    :param ndcg10: evaluation metric score NDCG@10
    :param p3: evaluation metric score P@3
    :param p5: evaluation metric score P@5
    :param p10: evaluation metric score P@10
    :param _map: evaluation metric score MAP
    """
    def __init__(self, method="", match_field="", rank_field="", loss_type="", query_type="", neg_type="", 
                 ndcg3=0, ndcg5=0, ndcg10=0, p3=0, p5=0, p10=0, _map=0):
        
        self.method = method
        self.match_field = match_field
        self.rank_field = rank_field
        self.loss_type = loss_type
        self.query_type = query_type
        self.neg_type = neg_type
        self.ndcg3  = ndcg3
        self.ndcg5  = ndcg5
        self.ndcg10 = ndcg10
        self.p3     = p3
        self.p5     = p5
        self.p10    = p10
        self._map   =  _map
        
    def __repr__(self):
        return {
            "Method"            : self.method.capitalize(),
            "Matching Field"    : self.match_field,
            "Ranking Field"     : self.rank_field,
            "Loss"              : self.loss_type,
            "Training Data"     : self.query_type,
            "Negative Sampling" : self.neg_type,
            "NDCG@3"            : "{0:.4f}".format(self.ndcg3),
            "NDCG@5"            : "{0:.4f}".format(self.ndcg5),
            "NDCG@10"           : "{0:.4f}".format(self.ndcg10),
            "P@3"               : "{0:.4f}".format(self.p3),
            "P@5"               : "{0:.4f}".format(self.p5),
            "P@10"              : "{0:.4f}".format(self.p10),
            "MAP"               : "{0:.4f}".format(self._map)
        }

class Evaluation(object):
    """ Class for generating evaluation of re-ranked results """

    rankers     = ["unsupervised", "supervised"] 
    rank_fields = ["BERT-Q-a", "BERT-Q-q"]
    loss_types  = ["triplet", "softmax"]
    query_types = ["faq", "user_query"]
    neg_types   = ["simple", "hard"]
    top_k       = [3, 5, 10]

    def get_eval_output(self, rank_results_filepath):
        """ Generate evaluation metrics and save them into a dictionary
        
        :param rank_results_filepath: rank results filepath
        :return: Python dictionary
        """
        output = dict()
        output['eval'] = dict()

        for ranker in Evaluation.rankers:

            # Compute metrics for the unsupervised method
            if ranker == "unsupervised":
                file_path = rank_results_filepath + "/" + ranker

                method = ranker

                answer_metric = []
                question_metric = []
                question_answer_metric = []
                question_answer_concat_metric = []

                for k in Evaluation.top_k:
                    # compute NDCG@k
                    answer_metric.append(compute_ndcg(file_path + "/es_query_by_answer.json", k))
                    question_metric.append(compute_ndcg(file_path + "/es_query_by_question.json", k))
                    question_answer_metric.append(compute_ndcg(file_path + "/es_query_by_question_answer.json", k))
                    question_answer_concat_metric.append(compute_ndcg(file_path + "/es_query_by_question_answer_concat.json", k))

                    # compute P@k
                    answer_metric.append(compute_prec(file_path + "/es_query_by_answer.json", k))
                    question_metric.append(compute_prec(file_path + "/es_query_by_question.json", k))
                    question_answer_metric.append(compute_prec(file_path + "/es_query_by_question_answer.json", k))
                    question_answer_concat_metric.append(compute_prec(file_path + "/es_query_by_question_answer_concat.json", k))

                # compute MAP
                answer_metric.append(compute_map(file_path + "/es_query_by_answer.json"))
                question_metric.append(compute_map(file_path + "/es_query_by_question.json"))
                question_answer_metric.append(compute_map(file_path + "/es_query_by_question_answer.json"))
                question_answer_concat_metric.append(compute_map(file_path + "/es_query_by_question_answer_concat.json"))
                
                result = Result(method="unsupervised", match_field="answer", 
                    ndcg3=answer_metric[0], ndcg5=answer_metric[1], ndcg10=answer_metric[2], 
                    p3=answer_metric[3], p5=answer_metric[4], p10=answer_metric[5], _map=answer_metric[6]
                )
                
                output['eval'][method + "_answer"] = result.__repr__()
                
                result = Result(method="unsupervised", match_field="question",
                    ndcg3=question_metric[0], ndcg5=question_metric[1], ndcg10=question_metric[2], 
                    p3=question_metric[3], p5=question_metric[4], p10=question_metric[5], _map=question_metric[6]
                )
                
                output['eval'][method + "_question"] = result.__repr__()
                
                result = Result(method="unsupervised", match_field="question_answer",
                    ndcg3=question_answer_metric[0], ndcg5=question_answer_metric[1], ndcg10=question_answer_metric[2], 
                    p3=question_answer_metric[3], p5=question_answer_metric[4], p10=question_answer_metric[5], _map=question_answer_metric[6]
                )
                
                output['eval'][method + "_question_answer"] = result.__repr__()
                            
                result = Result(method="unsupervised", match_field="question_answer_concat",
                    ndcg3=question_answer_concat_metric[0], ndcg5=question_answer_concat_metric[1], ndcg10=question_answer_concat_metric[2], 
                    p3=question_answer_concat_metric[3], p5=question_answer_concat_metric[4], p10=question_answer_concat_metric[5], _map=question_answer_concat_metric[6]
                )
                
                output['eval'][method + "_question_answer_concat"] = result.__repr__()
                
            elif ranker == 'supervised':
            
            # Compute metrics for the supervised method
                
                for rank_field in Evaluation.rank_fields:
                
                    for loss_type in Evaluation.loss_types:

                        for query_type in Evaluation.query_types:

                            for neg_type in Evaluation.neg_types:

                                file_path = rank_results_filepath + "/" + ranker + "/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type

                                if os.path.isdir(file_path):

                                    method = loss_type + "_" + neg_type + "_" + query_type + "_" + rank_field

                                    answer_metric = []
                                    question_metric = []
                                    question_answer_metric = []
                                    question_answer_concat_metric = []

                                    for k in Evaluation.top_k:

                                        # compute NDCG@k
                                        answer_metric.append(compute_ndcg(file_path + "/reranked_query_by_answer.json", k))
                                        question_metric.append(compute_ndcg(file_path + "/reranked_query_by_question.json", k))
                                        question_answer_metric.append(compute_ndcg(file_path + "/reranked_query_by_question_answer.json", k))
                                        question_answer_concat_metric.append(compute_ndcg(file_path + "/reranked_query_by_question_answer_concat.json", k))

                                        # compute P@k
                                        answer_metric.append(compute_prec(file_path + "/reranked_query_by_answer.json", k))
                                        question_metric.append(compute_prec(file_path + "/reranked_query_by_question.json", k))
                                        question_answer_metric.append(compute_prec(file_path + "/reranked_query_by_question_answer.json", k))
                                        question_answer_concat_metric.append(compute_prec(file_path + "/reranked_query_by_question_answer_concat.json", k))

                                    # compute map
                                    answer_metric.append(compute_map(file_path + "/reranked_query_by_answer.json"))
                                    question_metric.append(compute_map(file_path + "/reranked_query_by_question.json"))
                                    question_answer_metric.append(compute_map(file_path + "/reranked_query_by_question_answer.json"))
                                    question_answer_concat_metric.append(compute_map(file_path + "/reranked_query_by_question_answer_concat.json"))
                                    
                                    result = Result(
                                        method="supervised", match_field="answer", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg3=answer_metric[0], ndcg5=answer_metric[1], ndcg10=answer_metric[2], 
                                        p3=answer_metric[3], p5=answer_metric[4], p10=answer_metric[5], _map=answer_metric[6]
                                    )

                                    output['eval'][method + "_answer"] = result.__repr__()
    
                                    result = Result(
                                        method="supervised", match_field="question", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg3=question_metric[0], ndcg5=question_metric[1], ndcg10=question_metric[2], 
                                        p3=question_metric[3], p5=question_metric[4], p10=question_metric[5], _map=question_metric[6]
                                    )

                                    output['eval'][method + "_question"] = result.__repr__()

                                    result = Result(
                                        method="supervised", match_field="question_answer", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg3=question_answer_metric[0], ndcg5=question_answer_metric[1], ndcg10=question_answer_metric[2], 
                                        p3=question_answer_metric[3], p5=question_answer_metric[4], p10=question_answer_metric[5], _map=question_answer_metric[6]
                                    )

                                    output['eval'][method + "_question_answer"] = result.__repr__()
                                    
                                    result = Result(
                                        method="supervised", match_field="question_answer_concat", rank_field=rank_field, loss_type=loss_type, query_type=query_type, neg_type=neg_type,
                                        ndcg3=question_answer_concat_metric[0], ndcg5=question_answer_concat_metric[1], ndcg10=question_answer_concat_metric[2], 
                                        p3=question_answer_concat_metric[3], p5=question_answer_concat_metric[4], p10=question_answer_concat_metric[5], _map=question_answer_concat_metric[6]
                                    )
                                    
                                    output['eval'][method + "_question_answer_concat"] = result.__repr__()
                                    
                                    
        
        return output

    def get_eval_df(self, rank_results_filepath):
        """ Generate evaluation DataFrame from rank_results filepath 
        
        :param rank_results_filepath: rank results filepath
        :return: evaluation DataFrame
        """
        output = self.get_eval_output(rank_results_filepath)
        df = pd.DataFrame.from_dict({(i,j): output[i][j] 
                                                for i in output.keys() 
                                                    for j in output[i].keys()},
                                                                orient='index')

        df.reset_index(drop=True, inplace=True)
        return df
    
