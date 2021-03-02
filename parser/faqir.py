from shared.utils import dump_to_json
import pandas as pd
import numpy as np
import xmltodict
import json

class FAQIR_XML_Parser(object):
    """ Class for parsing & extracting data from FAQIRv1.0.xml file """
    
    def __init__(self):
        self.data = dict()
        self.queries = []
        self.num_queries = 0
        self.qa_pairs = []
        self.num_qa_pairs = 0
        self.ircandidates = []
        self.num_ircandidates = 0
        self.query_answer_pairs = []
        self.num_query_answer_pairs = 0
        self.qa_pairs_df = None
        self.queries_df = None
        self.ircandidates_df = None
        self.query_answer_pairs_df = None

    def get_data(self, xml_file):
        """ Get data from XML file 
        
        :param xml_file: input XML file
        :return: collections.OrderedDict type 
        """
        data = None
        with open(xml_file) as file:
            data = xmltodict.parse(file.read())
        return data

    def to_dict(self, data):
        """ Convert data to dictionary type 
        
        :param data: collections.OrderedDict type
        :return: Python dictionary
        """
        # convert data to str representation
        data = json.dumps(data)
        # convert data to dict representation
        data = json.loads(data)
        return dict(data)

    def extract_queries(self, data):
        """ Extract list of available queries
        
        :param data: parsed XML data into dictionary
        :return: query list
        """
        # get queries
        queries = data["IRSet"]["queries"]["Query"]
        
        # loop through all queries, parse and store data to a final list
        query_data = []
        for query in queries:
            query = dict(query)
            expansions = None
            if 'Expansions' in query:
                expansions = query['Expansions']
            queryGroupID = 0
            if 'QueryGroupID' in query:
                queryGroupID = query['QueryGroupID']
            author = None 
            if 'Author' in query:
                author = query['Author']
            original = None
            if 'original' in query:
                original = query['original']
            infneed = None
            if 'infneed' in query:
                infneed = str(query['infneed'])
            exwords = None
            if 'exWords' in query:
                if query['exWords'] is not None:
                    exwords = list(query['exWords']['string'])
            query_string = None
            if 'QueryString' in query:
                query_string = query['QueryString']
            rel_docs = None
            if 'relDocs' in query:
                rel_docs = query['relDocs']
            fold = None
            if 'Fold' in query:
                fold = query['Fold']
            
            data = {
                "queryGroupID": queryGroupID,
                "expansions": expansions,
                "author": author,
                "question": original,
                "infneed": infneed,
                "exwords": exwords,
                "query_string": query_string,
                "rel_docs": rel_docs,
                "fold": fold
            }
            
            query_data.append(data)
        return query_data

    def extract_ircandidates(self, data):
        """ Extract list of available queries
        
        :param data: parsed XML data into dictionary
        :return: query list
        """
        irCandidateList = data["IRSet"]["relCandidates"]["IRCandidateList"]
        
        # loop through all queries, parse and store data to a final list
        ircandidates = []
        for elem in irCandidateList:
            elem = dict(elem)
            irCandidate = elem['candidates']['IRCandidate']
            queryGroupID = elem['grpId']

            for irc in irCandidate:
                id = irc['Id']
                annotation = irc['Annotations']['Annotation']
                data = {
                    "id": id, 
                    "queryGroupID": queryGroupID,
                    "annotation": annotation
                }
                ircandidates.append(data)

        return ircandidates

    def extract_qa_pairs(self, data):
        """ Extract qa from data 
        
        :param data: parsed XML data into dictionary
        :return: question-answer pairs
        """
        qaPair = data["IRSet"]["KB"]["Pairs"]["qaPair"]
    
        # loop through all queries, parse and store data to a final list
        qa_pairs = []
        i = 0
        for pair in qaPair:
            i += 1
            pair = dict(pair)
            
            id = ""
            if 'Id' in pair:
                id = pair["Id"]
            question = ""
            if 'Question' in pair:
                question = pair['Question']
            answer = ""
            if 'Answer' in pair:
                answer = pair['Answer']
            
            data = {
                'id': id,
                'question': question,
                'answer': answer,
                'type': 'original'
            }
        
            qa_pairs.append(data) 
            
        return qa_pairs
    
    def extract_label(self, data):
        """ Extract label value from ircandidate lists"""
        label = 0
        
        if isinstance(data, dict):
            val = data['Val']
            val = int(val)
            
            if val == 1:
                label = 1
            else:
                label = 0
                
        elif isinstance(data, list):
            values = [int(item['Val']) for item in data]
            
            if 1 in values:
                label = 1
            else:
                label = 0
        
        else:
            pass
        
        return label

    def process_qa_pairs(self, qa_pairs):
        """ Process list of qa pairs and generate qa pairs dataframes
        
        :param qa_pairs: qa pair list
        :return: qa_pairs dataframes
        """
        qa_pairs_df = pd.DataFrame.from_records(qa_pairs)
        qa_pairs_df_copy = qa_pairs_df[['id', 'answer']]
        qa_pairs_df.rename(columns={'type': 'query_type'}, inplace=True)
        qa_pairs_df['query_type'] = 'faq'
        qa_pairs_df['label'] = 1
        return qa_pairs_df, qa_pairs_df_copy

    def process_ircandidates(self, ircandidates):
        """ Process list of ircandidates and generate ircandidate dataframe
        
        :param ircandidates: ircandidate list
        :return: ircandidate dataframe
        """
        ircandidates_df = pd.DataFrame.from_records(ircandidates)
        ircandidates_df['label'] = ircandidates_df['annotation'].apply(self.extract_label)
        return ircandidates_df

    def process_queries(self, queries):
        """ Process list of queries and generate queries dataframe
        
        :param queries: question-paraphrase list
        :return: queries dataframe
        """
        queries_df = pd.DataFrame.from_records(queries)
        queries_df = queries_df[queries_df['question'] != 'temp']
        return queries_df

    def get_user_query_pair_df(self, faq_qa_pair_df, queries_df):
        """ Generate user queries dataframe
        
        :param faq_qa_pair_df: faq qa pair dataframe
        :param queries_df: queries dataframe
        :return: user query dataframe
        """
        # join faq_qa_pair_df with queries_df on 'queryGroupID' field
        user_query_pair_df = pd.merge(faq_qa_pair_df, queries_df, on=['queryGroupID'])
        
        # select only positives
        user_query_pair_df = user_query_pair_df[user_query_pair_df['label'] == 1]
        user_query_pair_df = user_query_pair_df[['answer', 'label', 'query_string']]
        user_query_pair_df.rename(columns={'query_string': 'question'}, inplace=True)
        user_query_pair_df['query_type'] = 'user_query'
        return user_query_pair_df

    def get_query_answer_pairs_df(self, qa_pairs_df, user_query_pair_df):
        """ Generate dataframe consisting faq & user queries 
        
        :param qa_pairs_df: qa dataframe
        :param user_query_pair_df: user query dataframe
        :return: qa_pair_df + user_query_pair_df
        """
        query_answer_pairs_df = pd.concat([qa_pairs_df, user_query_pair_df])
        query_answer_pairs_df.index = np.arange(1, len(query_answer_pairs_df)+1)
        query_answer_pairs_df['id'] = query_answer_pairs_df.index
        return query_answer_pairs_df

    def extract_data(self, xml_file):
        """ Parse & extract data from XML file 
        
        :param xml_file: XML file
        """
        # read xml file into dictionary
        data = self.get_data(xml_file)
        data = self.to_dict(data)

        # extract qa_pairs, queries, ircandidates
        qa_pairs = self.extract_qa_pairs(data)
        queries = self.extract_queries(data)
        ircandidates = self.extract_ircandidates(data)
  
        # generate dataframes
        qa_pairs_df, qa_pairs_df_copy = self.process_qa_pairs(qa_pairs)
        ircandidates_df = self.process_ircandidates(ircandidates)
        queries_df = self.process_queries(queries)

        # next, join qa_pairs_df_copy with ircandidates_df on 'id' field
        faq_qa_pair_df = pd.merge(qa_pairs_df_copy, ircandidates_df, on=['id'])
        queries_df.drop(['question'], axis=1, inplace=True)
        
        user_query_pair_df = self.get_user_query_pair_df(faq_qa_pair_df, queries_df)
        query_answer_pairs_df = self.get_query_answer_pairs_df(qa_pairs_df, user_query_pair_df)
        query_answer_pairs = list(query_answer_pairs_df.T.to_dict().values())

        self.data = data
        self.qa_pairs = qa_pairs
        self.num_qa_pairs = len(qa_pairs)
        self.queries = queries
        self.num_queries = len(queries)
        self.ircandidates = ircandidates
        self.num_ircandidates = len(ircandidates)
        self.query_answer_pairs = query_answer_pairs
        self.num_query_answer_pairs = len(query_answer_pairs)
        self.qa_pairs_df = qa_pairs_df
        self.queries_df = queries_df
        self.ircandidates_df = ircandidates_df
        self.query_answer_pairs_df = query_answer_pairs_df


