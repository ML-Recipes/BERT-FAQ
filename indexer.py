from elasticsearch_dsl import Index, Document, Integer, Text, analyzer, Keyword, Double
from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch, helpers
from evaluation import get_relevance_label_df
from tqdm import tqdm
import logging
import json
import os

class QA(Document):
    id = Integer()
    question = Text()
    answer = Text()
    question_answer = Text()

def ingest_data(data, es, index):
    """ Ingest data as a bulk of documents to ES index """

    try:
        docs = []
        for pair in tqdm(data):
            
            # initialize QA document
            doc = QA()

            if 'id' in pair:
                doc.id = pair['id']
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

def get_faq_qa_pairs(query_answer_pairs_filepath):
    """ Get faq qa pair list """
    relevance_label_df = get_relevance_label_df(query_answer_pairs_filepath)
    faq_qa_pair_df = relevance_label_df[relevance_label_df['query_type'] == 'faq']
    faq_qa_pairs = faq_qa_pair_df.T.to_dict().values()
    faq_qa_pairs = list(faq_qa_pairs)
    return faq_qa_pairs

if __name__ == "__main__":
    try:

        # Ingesting data to Elasticsearch
        es = connections.create_connection(hosts=['localhost'])
        
        dirnames = ["FAQIR", "StackFAQ"]
        
        index_name = ""
        faq_qa_pairs = []
        for dirname in dirnames:
            if dirname == "FAQIR":
                index_name = "faqir"
                filepath = 'data/' + dirname + '/query_answer_pairs.json'
                faq_qa_pairs = get_faq_qa_pairs(filepath)
            else:
                index_name = "stackfaq"
                filepath = 'data/' + dirname + '/query_answer_pairs.json'
                faq_qa_pairs = get_faq_qa_pairs(filepath)

            print("{} records: ".format(dirname), len(faq_qa_pairs))

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
            ingest_data(faq_qa_pairs, es, index_name)

            print("Finished indexing {} records to {} index".format(len(faq_qa_pairs), index_name))

    except Exception:
        logging.error('exception occured', exc_info=True)
