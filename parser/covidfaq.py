

class CovidFAQ_Parser(object):
    """ Class for parsing & extracting data from aligned_question_answer.csv """

    def __init__(self):
        self.faq_pairs = []
        self.num_faq_pairs = 0
        self.user_query_pairs = []
        self.num_user_query_pairs = 0
        self.query_answer_pairs = []
        self.num_query_answer_pairs = 0
        
    def extract_pairs(self, df, query_type):
        """ Extract qa pairs from DataFrame for a given query_type
    
        :param df: input DataFrame
        :param query_type: faq or user_query
        :return: qa pairs
        """
        qa_pairs = []
        if query_type == "faq":
            # select question, answer columns
            df = df[['question', 'answer']]

            for _, row in df.iterrows():
                data = dict()
                data["label"] = 1
                data["query_type"] = "faq"
                data["question"] = row["question"]
                data["answer"] = row["answer"]
                qa_pairs.append(data)

        elif query_type == "user_query":
            # select query_string, answer columns
            df = df[['query_string', 'answer']]

            for _, row in df.iterrows():
                data = dict()
                data["label"] = 1
                data["query_type"] = "user_query"
                data["question"] = row["query_string"]
                data["answer"] = row["answer"]
                qa_pairs.append(data)
        else:
            raise ValueError('error, no query_type found for {}'.format(query_type))

        # remove duplicates
        pairs = []
        for pair in qa_pairs:
            if pair not in pairs:
                pairs.append(pair)

        return pairs

    def get_query_answer_pairs(self, faq_pairs, user_query_pairs):
        """ Generate query answer pair list using faq, user query pairs 
        
        :param faq_pairs: faq pairs
        :param user_query_pairs: user query pairs
        :return: query answer pairs
        """
        query_answer_pairs = faq_pairs + user_query_pairs
        
        # assign id to each query_answer_pair
        i = 0
        for item in query_answer_pairs:
            i += 1
            item.update({"id": str(i)})
        
        return query_answer_pairs

            
    def extract_data(self, df):
        """ Extract data from DataFrame
        
        :param df: Pandas DataFrame
        """
        # extract faq_pairs, user_pairs
        faq_pairs = self.extract_pairs(df, query_type='faq')
        user_query_pairs = self.extract_pairs(df, query_type='user_query')
        query_answer_pairs = self.get_query_answer_pairs(faq_pairs, user_query_pairs)
        
        self.faq_pairs = faq_pairs
        self.num_faq_pairs = len(faq_pairs)
        self.user_query_pairs = user_query_pairs
        self.num_user_query_pairs = len(user_query_pairs)
        self.query_answer_pairs = query_answer_pairs
        self.num_query_answer_pairs = len(query_answer_pairs)

        