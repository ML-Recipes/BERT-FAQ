from elasticsearch_dsl import Document, Text, analyzer, Integer, Keyword, Double
import xmltodict
import json

class StackFAQ_XML_Parser(object):
    """ Class for parsing & extracting data from stackExchange-FAQ.xml file """
    
    def __init__(self):
        self.data = dict()
        self.query_answer_pairs = []
        self.num_query_answer_pairs = 0
        
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
    
    def extract_query_answer_pairs(self, data):
        """ Extract a list of questions for a given answer 
        
        :param data: json object
        :return: list of query-answer pairs
        """
        # get all question-answer pairs
        qa_pair = data["root"]["qapair"]
        
        # loop through all queries, parse and store data to a final list
        query_answer_pairs = []
        i = 0
        for qa in qa_pair:
            qa = dict(qa)

            rephr = []
            if 'rephr' in qa:
                rephr = qa['rephr']
            
            question = ""
            if 'question' in qa:
                question = qa['question']
            
            answers = []
            if 'answer' in qa:
                answers = qa['answer']
            
            if answers:
                if isinstance(answers, list):
                    for a in answers:
                        if isinstance(rephr, list):
                            for r in rephr:
                                if r:
                                    if r != "*":
                                        i += 1
                                        data = {
                                            "id": str(i),
                                            "question": r,
                                            "answer": a,
                                            "query_type": "user_query",
                                            "label": 1
                                        }
                                        query_answer_pairs.append(data)
                        
                        i += 1
                        data = {
                            "id": str(i),
                            "question": question,
                            "answer": a,
                            "query_type": "faq",
                            "label": 1
                        }
                        query_answer_pairs.append(data)
                else:
                    if isinstance(rephr, list):
                        for r in rephr:
                            if r:
                                if r != "*":
                                    i += 1
                                    data = {
                                        "id": str(i),
                                        "question": r,
                                        "answer": answers,
                                        "query_type": "user_query",
                                        "label": 1
                                    }
                                    query_answer_pairs.append(data)
                    
                    
                    i += 1
                    data = {
                        "id": str(i),
                        "question": question,
                        "answer": answers,
                        "query_type": "faq",
                        "label": 1
                    }
                    query_answer_pairs.append(data)
                                
        return query_answer_pairs

    def extract_data(self, xml_file):
        """ Extract data from XML file
        
        :param xml_file: XML file
        """
        # read xml file
        data = self.get_data(xml_file)
        # convert data to dictionary
        self.data = self.to_dict(data)
        
        # extract query_answer_pairs 
        self.query_answer_pairs = self.extract_query_answer_pairs(self.data)
        self.num_query_answer_pairs = len(self.query_answer_pairs)
