import logging

class History_Searcher(object):
    """ Class for retrieving Elasticsearch documents over time
    :param es: Elasticsearch instance
    :param index: Elasticsearch index name
    :param fields: query fields
        if fields=None retrieve all results from index
    :param top_k: Elasticsearch top-k results.
        if top_k=None retrieve all results; else retrieve top-k results
    """
    def __init__(self, es, index, fields=None, top_k=None):
        self.es = es
        self.index = index
        self.fields = fields
        self.top_k = top_k
        self.total_hits = 0
        self.max_score = 0
        self.results = []
    def query(self, query_string):
        """ Query ES index and retrive documents
        :param query_string: query string
        :return: ES results
        """
        try:
            response = None
            if self.fields is None or self.top_k is None:
                response = self.es.search(
                    index=self.index,
                    body={
                        "query": {
                            "multi_match": {
                                "query": query_string
                            }
                        }
                    }
                )
            else:
                response = self.es.search(
                    index=self.index,
                    body={
                        "size": self.top_k,
                        "query": {
                            "multi_match": {
                                "query": query_string,
                                "fields": self.fields
                            }
                        }
                    }
                )
            hits = response['hits']['hits']
            max_score = response['hits']['max_score']
            total_hits = response['hits']['total']['value']
            results = []
            for hit in hits:
                score = hit['_score']
                norm_score = score / max_score
                question = hit['_source']['question']
                answer = hit['_source']['answer']
                question_answer = hit['_source']['question_answer']
                sourceUrl = hit['_source']['sourceUrl']
                sourceName = hit['_source']['sourceName']
                date =  hit['_source']['date']
                month =  hit['_source']['month']
                
                results.append(
                    {
                        "score": norm_score, "question": question,
                        "answer": answer, "question_answer": question_answer,
                        "sourceUrl": sourceUrl, "sourceName": sourceName,
                        "date": date, "month": month
                    }
                )
            self.results = results
            self.max_score = max_score
            self.total_hits = total_hits
        except Exception:
            logging.error('exception occured', exc_info=True)
        return self.results
