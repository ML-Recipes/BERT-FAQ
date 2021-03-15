from faq_bert import FAQ_BERT
from searcher import Searcher

class FAQ_BERT_Ranker(object):
    """ Class to generate top-k ranked results for a given input query string 
    
    :param es: Elasticsearch instance
    :param index: Elasticsearch index name
    :param top_k: parameter used during model training (e.g top_k=100)
    :param bert_model_path: bert model path
    """
    def __init__(self, es, index, fields, top_k, bert_model_path):
        self.es = es
        self.index = index
        self.fields = fields
        self.top_k = top_k
        self.bert_model_path = bert_model_path
        self.searcher = Searcher(es, index, fields, top_k)

        self.es_topk_results = []
        self.bert_topk_preds = []
        self.ranked_results  = []

    def get_es_topk_results(self, query_string):
        """ Get Elasticsearch top-k results 
        
        :param query_string: Elasticsearch query string
        :return: ES top-k results
        """
        results = self.searcher.query(query_string)
        es_topk_results = [
            {
                'es_score': float("{0:.4f}".format(doc['score'])), 
                'question': query_string, 
                'answer': doc['answer']
            } for doc in results
        ]
        
        return es_topk_results

    def get_bert_topk_preds(self, es_topk_results):
        """ Get BERT top-k predictions 
        
        :param es_topk_results: ES top-k results
        :return: BERT predictions on ES top-k results
        """
        faq_bert = FAQ_BERT(bert_model_path=self.bert_model_path)

        bert_topk_preds = [
            {
                'question': doc['question'], 
                'answer': doc['answer'], 
                'es_score': doc['es_score'],
                'bert_score': float("{0:.4f}".format(faq_bert.predict(doc['question'], doc['answer'])))
            } for doc in es_topk_results
        ]
        
        return bert_topk_preds

    def get_ranked_results(self, bert_topk_preds):
        """ Get top-k re-ranked results 
        
        :param bert_topk_preds: bert top-k results
        :return: ranked list of top-k results in descending order by score
        """
        norm_results = [
            {
                'question': doc['question'], 
                'answer': doc['answer'], 
                'es_score': doc['es_score'],
                'bert_score': doc['bert_score'],
                'score': float("{0:.4f}".format(doc['es_score'] + doc['bert_score']))
            } for doc in bert_topk_preds
        ]

        ranked_results = sorted(norm_results, key=lambda x: x['score'], reverse=True)
        
        return ranked_results

    def rank_results(self, query_string):
        """ Rank ES top-k results for a given input query string 
            using BERT pretrained model
        
        :param query_string: input query
        :return: ES top-k ranked results
        """
        es_topk_results = self.get_es_topk_results(query_string)
        bert_topk_preds = self.get_bert_topk_preds(es_topk_results)
        ranked_results = self.get_ranked_results(bert_topk_preds)

        self.es_topk_results = es_topk_results
        self.bert_topk_preds = bert_topk_preds
        self.ranked_results  = ranked_results
        
        return ranked_results


    