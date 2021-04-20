import datetime as dt
import pandas as pd
import json
import os
import time

import sys
sys.path.insert(0, "../BERT-FAQ")
print(sys.path)
from shared.utils import dump_to_pickle

class History_Parser(object):
    """ Class for parsing & extracting CovidFAQ data from .jsonl files """
    
    def __init__(self, path="./topic_classification/data/", snapshots = ["schema_v0.1", "schema_v0.2", "schema_v0.3"]):
        self.path = path
        self.snapshots = snapshots
        self.all_data = []
        self.train_data = []
        self.infer_data = []

        self.list_of_questions = []

    def get_all_questions(self, output_path):
        i = 0

        # Load data from different snapshots (i.e., stored in schema_v0.x folders)
        for snapshot in self.snapshots:
            input_path = self.path + snapshot + "/"
            for filename in os.listdir(input_path):
                if filename.endswith(".jsonl"): 
                    with open(input_path + filename, 'r') as json_file:
                        json_list = list(json_file)

                        for json_str in json_list:
                            result = json.loads(json_str)
                            
                            topics = []
                            if 'topic' in result:
                                if isinstance(result['topic'], list):
                                    topics = result['topic']
                                elif isinstance(result['topic'], str) and (len(result['topic']) > 0):
                                    l = [result['topic']]
                                    topics = l

                            sourceName = ""
                            if 'sourceName' in result:
                                sourceName = result['sourceName']
                            
                            sourceUrl = ""
                            if 'sourceUrl' in result:
                                sourceUrl = result['sourceUrl']

                            time = None
                            date = None
                            if 'dateLastChanged' in result:
                                time = result['dateLastChanged']
                                date = str(dt.datetime.utcfromtimestamp(time).strftime("%Y%m%d"))
                                month = str(dt.datetime.utcfromtimestamp(time).strftime("%Y-%m"))
                            elif 'dateScraped' in result:
                                time = result['dateScraped']
                                date = str(dt.datetime.utcfromtimestamp(time).strftime("%Y%m%d"))
                                month = str(dt.datetime.utcfromtimestamp(time).strftime("%Y-%m"))

                            wayBackUrl = ""
                            if date is not None and sourceUrl is not None:
                                wayBackUrl = "https://web.archive.org/web/" + date + "/" + sourceUrl

                            question = ""
                            if 'questionText' in result:
                                question = result['questionText']
                            elif 'question' in result:
                                question = result['question']

                            answer = ""
                            if 'answerText' in result:
                                answer = result['answerText']
                            elif 'answer' in result:
                                answer = result['answer']
                            
                            version = snapshot

                            self.list_of_questions.append(question)
                            i = i+1
                else:
                    # Skip parsing non json files
                    continue
            
            print(i)
            return self.list_of_questions

    def generate_train_test(self, output_path):

        # Read FAQ data with manually annotated topics
        filename = './data/CovidFAQ/historical_faqs_for_indexing.tsv'
        hist_data = pd.read_csv(filename, sep='\t', header=0)
        hist_data.drop(['wayBackUrl', 'dateScraped', 'date', 'dateStr', 'month'], axis=1, inplace=True)
        print(len(hist_data))
        
        # Drop duplicate values
        hist_data.drop_duplicates(keep=False, inplace=True)
        print(len(hist_data))

        hist_data = hist_data.rename(columns={"topic": "manualTopic"})
        print(hist_data)

        # Load data from different snapshots (i.e., stored in schema_v0.x folders)
        for snapshot in self.snapshots:
            input_path = self.path + snapshot + "/"
            for filename in os.listdir(input_path):
                if filename.endswith(".jsonl"): 
                    with open(input_path + filename, 'r') as json_file:
                        json_list = list(json_file)

                        for json_str in json_list:
                            result = json.loads(json_str)
                            
                            topics = []
                            if 'topic' in result:
                                if isinstance(result['topic'], list):
                                    topics = result['topic']
                                elif isinstance(result['topic'], str) and (len(result['topic']) > 0):
                                    l = [result['topic']]
                                    topics = l

                            sourceName = ""
                            if 'sourceName' in result:
                                sourceName = result['sourceName']
                            
                            sourceUrl = ""
                            if 'sourceUrl' in result:
                                sourceUrl = result['sourceUrl']

                            time = None
                            date = None
                            if 'dateLastChanged' in result:
                                time = result['dateLastChanged']
                                date = str(dt.datetime.utcfromtimestamp(time).strftime("%Y%m%d"))
                                month = str(dt.datetime.utcfromtimestamp(time).strftime("%Y-%m"))
                            elif 'dateScraped' in result:
                                time = result['dateScraped']
                                date = str(dt.datetime.utcfromtimestamp(time).strftime("%Y%m%d"))
                                month = str(dt.datetime.utcfromtimestamp(time).strftime("%Y-%m"))

                            wayBackUrl = ""
                            if date is not None and sourceUrl is not None:
                                wayBackUrl = "https://web.archive.org/web/" + date + "/" + sourceUrl

                            question = ""
                            if 'questionText' in result:
                                question = result['questionText']
                            elif 'question' in result:
                                question = result['question']

                            answer = ""
                            if 'answerText' in result:
                                answer = result['answerText']
                            elif 'answer' in result:
                                answer = result['answer']
                            
                            version = snapshot

                            self.all_data.append({
                                    'topics': topics,
                                    'sourceName': sourceName,
                                    'sourceUrl': sourceUrl,
                                    'wayBackUrl': wayBackUrl,
                                    'time': time,
                                    'date': date,
                                    'month': month,
                                    'question': question,
                                    'answer': answer,
                                    'version': version
                                })

                            if len(topics) > 0:
                                self.train_data.append({
                                    'topics': topics,
                                    'sourceName': sourceName,
                                    'sourceUrl': sourceUrl,
                                    'wayBackUrl': wayBackUrl,
                                    'time': time,
                                    'date': date,
                                    'month': month,
                                    'question': question,
                                    'answer': answer,
                                    'version': version
                                })
                            else:
                                self.infer_data.append({
                                    'topics': topics,
                                    'sourceName': sourceName,
                                    'sourceUrl': sourceUrl,
                                    'wayBackUrl': wayBackUrl,
                                    'time': time,
                                    'date': date,
                                    'month': month,
                                    'question': question,
                                    'answer': answer,
                                    'version': version
                                })
                else:
                    # Skip parsing non json files
                    continue
            
            all_df = pd.DataFrame(self.all_data)
            all_df_ = pd.merge(all_df, hist_data, how="left", on=["sourceUrl", "sourceName", "question", "answer"])

            all_df.to_csv(output_path + "all.csv", sep="\t", index=False)
            all_df_.to_csv(output_path + "all_w_manual.csv", sep="\t", index=False)
            #print(train_df)

            train_df = pd.DataFrame(self.train_data)
            train_df_ = pd.merge(train_df, hist_data, how="left", on=["sourceUrl", "sourceName", "question", "answer"])

            train_df.to_csv(output_path + "train.csv", sep="\t", index=False)
            train_df_.to_csv(output_path + "train_w_manual.csv", sep="\t", index=False)
            #print(train_df)

            infer_df = pd.DataFrame(self.infer_data)
            infer_df_ = pd.merge(infer_df, hist_data, how="left", on=["sourceUrl", "sourceName", "question", "answer"])

            infer_df.to_csv(output_path + "infer.csv", sep="\t", index=False)
            infer_df_.to_csv(output_path + "infer_w_manual.csv", sep="\t", index=False)
            #print(infer_df)

if __name__ == "__main__":
    h = History_Parser()

    output_path = "./topic_classification/data/"
    #h.generate_train_test(output_path)
    
    questions = h.get_all_questions(output_path)
    print(len(questions))
    dump_to_pickle(questions, "./topic_classification/data/questions.pkl")
