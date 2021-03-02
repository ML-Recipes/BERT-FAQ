from evaluation import get_relevance_label_df
from evaluation import get_relevance_label
from shared.utils import load_from_json
from shared.utils import dump_to_json
from searcher import Searcher
from tqdm import tqdm
import pandas as pd
import random

class Hard_Negatives_Generator(object):
    def __init__(self, es, index, query_by, top_k=10, query_type='faq'):
        self.es = es
        self.index = index
        self.query_by = query_by
        self.top_k = top_k
        self.query_type = query_type

    def get_hard_negatives(self, relevance_label_df):
        """ Get a list of hard negative question-answer pairs """

        # define Searcher instance
        s = Searcher(self.es, index=self.index, fields=self.query_by , top_k=self.top_k)

        # Generate a dictionary where {key: query_string, value: list of answers}
        relevance_label = get_relevance_label(relevance_label_df)

        unique_questions = []
        
        if self.query_type == "faq":
            test_queries = relevance_label_df[relevance_label_df['query_type'] == self.query_type]
            unique_questions = test_queries.query_string.unique()
        else:
            unique_questions = relevance_label_df.query_string.unique()

        results = []
        for query_string in tqdm(unique_questions):

            # perform query using question as query_string
            topk_results = s.query(query_string=query_string)

            # get the list of actual answers
            answers = relevance_label[query_string]

            # obtain relevance label for each answer
            rank = 0
            for doc in topk_results:
                topk_answer = doc['answer']

                # check if the answer is a true answer
                label = 0
                
                if topk_answer in answers:
                    label = 1

                if label == 0:
                    rank += 1
                    data = dict()
                    data["query_string"] = query_string
                    data["neg_answer"] = doc["answer"]
                    data["question"] = doc["question"]
                    data["question_answer"] = doc["question_answer"]
                    data["score"] = doc["score"]
                    data["label"] = label
                    data["rank"] = rank
                    results.append(data) 

        return results
