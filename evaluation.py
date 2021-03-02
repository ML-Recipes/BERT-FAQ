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

def get_evaluation_df(rank_results_filepath):
    output = dict()
    metric_name = []

    i = 0

    rankers = ["unsupervised", "supervised"] 
    loss_type = ["triplet", "softmax"]
    query_types = ["faq", "user_query"]
    neg_types = ["simple", "hard"]

    topk = [3, 5, 10]

    for ranker in rankers:

        # Compute metrics for the unsupervised method
        if ranker == "unsupervised":
            file_path = rank_results_filepath + "/" + ranker

            method = ranker

            answer_metric = []
            question_metric = []
            question_answer_metric = []
            question_answer_concat_metric = []

            i += 1
            for k in topk:
                # compute NDCG@k
                if i == 1:
                    metric_name.append("ndcg@" + str(k))

                answer_metric.append(compute_ndcg(file_path + "/es_query_by_answer.json", k))
                question_metric.append(compute_ndcg(file_path + "/es_query_by_question.json", k))
                question_answer_metric.append(compute_ndcg(file_path + "/es_query_by_question_answer.json", k))
                question_answer_concat_metric.append(compute_ndcg(file_path + "/es_query_by_question_answer_concat.json", k))

                # compute P@k
                if i == 1:
                    metric_name.append("prec@" + str(k))

                answer_metric.append(compute_prec(file_path + "/es_query_by_answer.json", k))
                question_metric.append(compute_prec(file_path + "/es_query_by_question.json", k))
                question_answer_metric.append(compute_prec(file_path + "/es_query_by_question_answer.json", k))
                question_answer_concat_metric.append(compute_prec(file_path + "/es_query_by_question_answer_concat.json", k))

            # compute MAP
            if i == 1:
                metric_name.append("MAP" + str(k))

            answer_metric.append(compute_map(file_path + "/es_query_by_answer.json"))
            question_metric.append(compute_map(file_path + "/es_query_by_question.json"))
            question_answer_metric.append(compute_map(file_path + "/es_query_by_question_answer.json"))
            question_answer_concat_metric.append(compute_map(file_path + "/es_query_by_question_answer_concat.json"))
            

            output[method + "_answer"] = answer_metric
            output[method + "_question"] = question_metric
            output[method + "_question_answer"] = question_answer_metric
            output[method + "_question_answer_concat"] = question_answer_concat_metric

        else:
        # Compute metrics for the supervised method
            for loss in loss_type:
                for query_type in query_types:
                    
                    for neg_type in neg_types:
                        
                        file_path = rank_results_filepath + "/" + ranker + "/" + loss + "/" + query_type + "/" + neg_type
                        
                        if os.path.isdir(file_path):

                            method = loss + "_" + query_type + "_" + neg_type

                            answer_metric = []
                            question_metric = []
                            question_answer_metric = []
                            question_answer_concat_metric = []

                            for k in topk:

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

                            # compute MAP
                            answer_metric.append(compute_map(file_path + "/reranked_query_by_answer.json"))
                            question_metric.append(compute_map(file_path + "/reranked_query_by_question.json"))
                            question_answer_metric.append(compute_map(file_path + "/reranked_query_by_question_answer.json"))
                            question_answer_concat_metric.append(compute_map(file_path + "/reranked_query_by_question_answer_concat.json"))
                            

                            output[method + "_answer"] = answer_metric
                            output[method + "_question"] = question_metric
                            output[method + "_question_answer"] = question_answer_metric
                            output[method + "_question_answer_concat"] = question_answer_concat_metric
                        else:
                            pass
                        
    data = dict()
    methods = []
    for method in output:
        metrics = output[method]
        metrics = ["{0:.4f}".format(x) for x in metrics]
        data[method] = metrics
        methods.append(method)

    # Generate evaluation DataFrame 
    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ["NDCG@3", "NDCG@5","NDCG@10", "P@3",	"P@5", "P@10", "MAP"]
    df['Method'] = methods
    df = df[["Method", "NDCG@3", "NDCG@5","NDCG@10", "P@3",	"P@5", "P@10", "MAP"]]
    return df


if __name__ == "__main__":
    # FAQIR
    rank_results_filepath="data/FAQIR/rank_results"
    df = get_evaluation_df(rank_results_filepath)
    df.to_csv(rank_results_filepath + "/results.csv", index=False)

    # StackFAQ
    rank_results_filepath="data/StackFAQ/rank_results"
    df = get_evaluation_df(rank_results_filepath)
    df.to_csv(rank_results_filepath + "/results.csv", index=False)

    
