import random
import pandas as pd
from tqdm import tqdm
from shared.utils import make_dirs
from shared.utils import load_from_json
import sys

class Training_Data_Generator(object):
    """ Class for generating ground-truth dataset used for feature learning
    
    :param random_seed: parameter used for reproducibility
    :param num_samples: total number of negative samples
    :param neg_type: negative samples type (simple or hard)
    :param query_type: query type (faq or user_query) 
    :param loss_type: the loss type as method used for BERT Fine-tuning (softmax or triplet loss)
    :param hard_filepath: the absolut path to hard negatives filepath
    """
    def __init__(self, random_seed=5, num_samples=24, neg_type='simple', query_type='faq', 
                loss_type='triplet', hard_filepath=''):
        
        self.random_seed = random_seed
        self.num_samples = num_samples
        self.hard_filepath = hard_filepath
        self.neg_type = neg_type
    
        self.query_type = query_type
        self.loss_type = loss_type
        self.pos_labels = []
        self.neg_labels = []
        self.num_pos_labels = 0
        self.num_neg_labels = 0

        self.id2qa = dict()
        self.id2negids = dict()
        self.df = pd.DataFrame()
        self.seq_len_df = pd.DataFrame()
        self.df_pos = pd.DataFrame()
        self.df_neg = pd.DataFrame()
        
        if self.query_type == 'faq':
            self.hard_filepath = self.hard_filepath + "/hard_negatives_faq.json"
        elif self.query_type == "user_query":
            self.hard_filepath = self.hard_filepath + "/hard_negatives_user_query.json"
        else:
            raise ValueError('error, no query_type found for {}'.format(query_type))

    def generate_pos_labels(self, query_answer_pairs):
        """ Generate positive labels from qa pairs 
        
        :param qa_pairs: list of dicts
        :return: list of positive labels
        """
        qap_df = pd.DataFrame.from_records(query_answer_pairs)
        qap_by_query_type = qap_df[qap_df['query_type'] == self.query_type]
        
        pos_labels = []
        for _, row in qap_by_query_type.iterrows():
            id = row['id']
            qa_pair = {
                "id": id,
                "label": 1,
                "question": row['question'],
                "answer": row['answer'],
                "query_type": row['query_type']
            }
            pos_labels.append(qa_pair)
            
            self.id2qa[id] = (qa_pair['question'], qa_pair['answer'], qa_pair['query_type'])
            
        return pos_labels
    
    def get_id2negids(self, id2qa):
        """ Generate random negative sample ids for qa pairs 
        
        :param id2qa: dictionary (id: key, question-answer (tuple): value)
        :return: dictionary (id: key, neg_ids: value)
        """
        random.seed(self.random_seed)
        id2negids = dict()
        total_qa = len(id2qa)
        ids = id2qa.keys()
        for id, qa in id2qa.items():
            neg_ids = random.sample([x for x in ids if x != id and x !=0], self.num_samples)
            id2negids[id] = neg_ids
        return id2negids

    def generate_neg_labels(self, id2negids):
        """ Generate negative labels from id2negids 
        
        :param id2negids: dictionary (id: key, neg_ids: value)
        :return: list of negative labels as dictionaries
        """
        neg_labels = []
        for k, v  in id2negids.items():
            for id in v:
                neg_label = dict()
                neg_label['id'] = str(k)
                neg_label['question'] = self.id2qa[k][0]
                neg_answer = self.id2qa[id][1]
                neg_label['answer'] = neg_answer
                neg_label['label'] = 0
                neg_label['query_type'] = self.id2qa[id][2]
                neg_labels.append(neg_label)
        return neg_labels

    def get_seq_len_df(self, query_answer_pairs):
        """ Get sequence length in dataframe 
        
        """
        seq_len = []
        for qa in tqdm(query_answer_pairs):
            qa['q_len'] = len(qa['question'])
            qa['a_len'] = len(qa['answer'])
            seq_len.append(qa)
        
        seq_len_df = pd.DataFrame(seq_len)
        return seq_len_df
    
    def get_pos_neg_df(self, query_answer_pairs):
        """ Generate positive, negative dataframes """
        pos_df = None
        neg_df = None
        pos_labels = []
        neg_labels = []
        id2negids = dict()
        
        if self.loss_type == "triplet":
            if self.neg_type == "simple":
                pos_labels = self.generate_pos_labels(query_answer_pairs)
                id2negids = self.get_id2negids(self.id2qa)
                neg_labels = self.generate_neg_labels(id2negids)
                pos_df = pd.DataFrame(pos_labels)
                neg_df = pd.DataFrame(neg_labels)
            elif self.neg_type == "hard":
                neg_labels = load_from_json(self.hard_filepath)
                pos_labels = query_answer_pairs
                neg_df = pd.DataFrame.from_records(neg_labels)
                neg_df = neg_df[neg_df['rank'] <= self.num_samples]
                pos_df = pd.DataFrame.from_records(pos_labels)
        elif self.loss_type == "softmax":
            if self.neg_type == "simple":
                pos_labels = self.generate_pos_labels(query_answer_pairs)
                id2negids = self.get_id2negids(self.id2qa)
                neg_labels = self.generate_neg_labels(id2negids)
                pos_df = pd.DataFrame(pos_labels)
                neg_df = pd.DataFrame(neg_labels)
            elif self.neg_type == "hard":
                neg_labels = load_from_json(self.hard_filepath)
                pos_labels = query_answer_pairs
                neg_df = pd.DataFrame.from_records(neg_labels)
                neg_df = neg_df[neg_df['rank'] <= self.num_samples]
                pos_df = pd.DataFrame.from_records(pos_labels)
        else:
            raise ValueError("error, no neg_type found for".format(self.neg_type))

        self.id2negids = id2negids
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels
        self.num_pos_labels = len(pos_labels)
        self.num_neg_labels = len(neg_labels)

        return pos_df, neg_df

    def generate_triplet_dataset(self, query_answer_pairs, output_path):
        """ Generate ground-truth dataset for feature learning 

        :param qa_pairs: question-answer pair list
        :param output_path: output path name
        """
        # create directory structure
        output_path = output_path + "/dataset/" + self.loss_type + "/" + self.query_type
        make_dirs(output_path)

        df = None
        pos_df = None
        neg_df = None
    
        if self.loss_type == "triplet":
            if self.neg_type == "simple":
                # generate pos, neg dataframes
                pos_df, neg_df = self.get_pos_neg_df(query_answer_pairs)
                
                # rename colnames
                pos_df['positive'] = pos_df['answer']
                neg_df['negative'] = neg_df['answer']
            
                # drop columns answer, label, query_type
                pos_df.drop(['answer', 'label', 'query_type'], axis=1, inplace=True)
                neg_df.drop(['answer', 'label', 'query_type'], axis=1, inplace=True)

                # convert colname to data types
                pos_df['id'] = pos_df['id'].astype(int)
                pos_df['question'] = pos_df['question'].astype(str)
                neg_df['id'] = neg_df['id'].astype(int)
                neg_df['question'] = neg_df['question'].astype(str)
            
                df = pd.merge(pos_df, neg_df, on=['question', 'id'])
                df.drop(['id'], axis=1, inplace=True)

            elif self.neg_type == "hard":
                # generate pos, neg dataframes
                pos_df, neg_df = self.get_pos_neg_df(query_answer_pairs)
                if ('label' in pos_df.columns) and ('label' in neg_df.columns):
                    pos_df.drop(['label'], axis=1, inplace=True)
                    neg_df.drop(['label'], axis=1, inplace=True)
    
                pos_df.rename(columns={'answer': 'pos_answer', 'question': 'query_string'}, inplace=True)
                df = pd.merge(pos_df, neg_df, on=['query_string'])
                df.drop_duplicates(inplace=True)

                df.drop(['id', 'query_type', 'question', 'question_answer', 'rank', 'score'], axis=1, inplace=True)
                df.rename(columns={'query_string': 'question', 'pos_answer': 'positive', 'neg_answer': 'negative'}, inplace=True)
        elif self.loss_type == "softmax":
            if self.neg_type == "simple":
                # generate pos, neg dataframes
                pos_df, neg_df = self.get_pos_neg_df(query_answer_pairs)
                df = pd.concat([pos_df, neg_df])
                df.drop(['id', 'query_type'], axis=1, inplace=True)
                
            elif self.neg_type == "hard": 
                # generate pos, neg dataframes
                pos_df, neg_df = self.get_pos_neg_df(query_answer_pairs)
                pos_df.drop(['id', 'query_type'], axis=1, inplace=True)

                neg_df.drop(['question_answer', 'rank', 'score', 'question'], axis=1, inplace=True)
                neg_df.rename(columns={'query_string': 'question', 'neg_answer': 'answer'}, inplace=True)
                df = pd.concat([pos_df, neg_df])
        else:
            raise ValueError("error, no loss_type found for {}".format(self.loss_type))

        df.to_csv(output_path + "/" + self.neg_type + "_" + self.query_type + "_dataset.csv", index=False)

        self.pos_df = pos_df
        self.neg_df = neg_df
        self.df = df

       
