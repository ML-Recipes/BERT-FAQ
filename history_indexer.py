from elasticsearch_dsl import Index, Document, Integer, Text, analyzer, Keyword, Double
from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch, helpers
from evaluation import get_relevance_label_df
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import json
import os

class QA(Document):
    sourceUrl = Text()
    sourceName = Text()
    date = Text()
    month = Text()
    question = Text()
    answer = Text()
    question_answer = Text()

def ingest_history_data(data, es, index):
    """ Ingest data as a bulk of documents to ES index """

    try:
        docs = []
        for pair in tqdm(data):
            # initialize QA document
            doc = QA()

            if 'sourceUrl' in pair:
                doc.sourceUrl = pair['sourceUrl']
            if 'sourceName' in pair:
                doc.sourceName = pair['sourceName']
            if 'date' in pair:
                date = str(pair['date'])
                year = date[:4]
                month = date[4:6]
                day = date[-2:]
                doc.date = day + "/" + month + "/" + year
            if 'month' in pair:
                doc.month = pair['month']
            if 'question' in pair:
                doc.question = pair['question']
            if 'answer' in pair:
                doc.answer = pair['answer']
            if 'question' in pair and 'answer' in pair:
                doc.question_answer = pair['question'] + " " + pair['answer']

            docs.append(doc.to_dict(include_meta=False))

        # bulk indexing
        response = helpers.bulk(es, actions=docs, index=index, doc_type='doc')
            
    except Exception:
        logging.error('exception occured', exc_info=True)

def get_history_qa_pairs(filename):
    """ Get faq qa pair list """
    df = pd.read_csv(filename, sep='\t', header=0)
    months = np.sort(df.month.unique())

    faq_dfs = []
    for m in months:
        subset = df.loc[df['month'] <= m]
        print(m + "\t" + str(len(subset)))
        snapshot = {'month': m, 'data': subset}
        faq_dfs.append(snapshot)

    return faq_dfs

if __name__ == "__main__":
    
    try:

        # Ingesting data to Elasticsearch
        es = connections.create_connection(hosts=['localhost'], http_auth=('elastic', 'elastic'))
        
        dirnames = ["CovidFAQ"]

        index_name = ""
        faq_qa_pairs = []

        # Define a list of months to display in the timeline
        months = ['2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04']

        # Load history data
        filename = './data/CovidFAQ/historical_faqs_for_indexing.tsv'
        snapshots = get_history_qa_pairs(filename)

        faq_qa_pairs = None
        for m in months:
            for s in snapshots:
                if m == s['month']:
                    faq_qa_pair_df = s['data']
                    break
                
            index_name = "covidfaq_" + m
            faq_qa_pairs = faq_qa_pair_df.T.to_dict().values()
            faq_qa_pairs = list(faq_qa_pairs)
             
            print("{} records: ".format(index_name), len(faq_qa_pairs))

            # Initialize index (only perform once)
            index = Index(index_name)

            # Define custom settings
            index.settings(
                number_of_shards=1,
                number_of_replicas=0
            )

            # Delete the index, ignore if it doesn't exist
            index.delete(ignore=404)

            # Create the index in Elasticsearch
            index.create()

            # Register a document with the index
            index.document(QA)

            # Ingest data to Elasticsearch
            ingest_history_data(faq_qa_pairs, es, index_name)

            print("Finished indexing {} records to {} index".format(len(faq_qa_pairs), index_name))

    except Exception:
        logging.error('exception occured', exc_info=True)
