
import re
import textdistance
from evaluation import get_relevance_label_df

import sys
sys.path.append('../question_generator/')
from questiongenerator import QuestionGenerator

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

def remove_urls(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    return text

class SyntheticQueryGenerator(object):
    def __init__(self, model_dir=None, query_type='faq', min_conf_score=0.60, answer_style='sentences', num_questions=None):
        
        if query_type not in {'faq', 'user_query'}:
            raise ValueError('error, query_type not exists')
        
        self.query_type = query_type
        self.min_conf_score = min_conf_score
        self.answer_style = answer_style
        self.num_questions = num_questions
        self.qg = QuestionGenerator(model_dir=model_dir)
    
    def generate_synthetic_qas(self, text):
        qas = self.qg.generate(
            text, answer_style=self.answer_style, num_questions=self.num_questions
        )
        return qas
    
    def generate_synthetic_query_answer_pairs(self, query_answer_pair_filepath):
    
        relevance_label_df = get_relevance_label_df(query_answer_pair_filepath)
        
        if self.query_type == 'faq':
            relevance_label_df = relevance_label_df[relevance_label_df['query_type'] == 'faq']
        elif self.query_type == 'user_query':
            relevance_label_df = relevance_label_df[relevance_label_df['query_type'] == 'user_query']
        
        synthetic_query_answer_pairs = []

        for _, row in relevance_label_df.iterrows():
            
            answer = row['answer']
            question = row['question']
            label = row['label']
            _id = row['id']

            answer = remove_urls(answer)
            t5_qas = self.generate_synthetic_qas(answer)
    
            t5_questions = [item['question'] for item in t5_qas if item['confidence'] >= self.min_conf_score]
            t5_questions = list(set(t5_questions))

            if t5_questions:
                for t5_question in t5_questions:
                    jc_sim = jaccard_similarity(question, t5_question)
                    lv_dist = textdistance.levenshtein.normalized_similarity(question, t5_question)

                    data = dict()
                    data['label'] = label
                    data['query_type'] = "synthetic"
                    data['org_question'] = question
                    data['question'] = t5_question
                    data['answer'] = answer
                    data['jc_sim'] = "{0:.4f}".format(jc_sim)
                    data['lv_dist'] = "{0:.4f}".format(lv_dist)
                    data['id'] = _id
                    
                    synthetic_query_answer_pairs.append(data)
            
        return synthetic_query_answer_pairs

