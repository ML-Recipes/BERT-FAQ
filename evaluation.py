from shared.utils import load_from_json
from tqdm import tqdm
import pandas as pd
import numpy as np

def get_relevance_label_df(query_answer_pair_filepath):
    query_answer_pair = load_from_json(query_answer_pair_filepath)
    relevance_label_df = pd.DataFrame.from_records(query_answer_pair)
    return relevance_label_df

def get_relevance_label(relevance_label_df):
    relevance_label_df.rename(columns={'question': 'query_string'}, inplace=True)
    relevance_label = relevance_label_df.groupby(['query_string'])['answer'].apply(list).to_dict()
    return relevance_label

