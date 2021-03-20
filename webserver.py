from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch, TransportError
from flask import Flask, request
from flask_cors import CORS
import json

from faq_bert_ranker import FAQ_BERT_Ranker
from shared.utils import isDir

try:
    es = connections.create_connection(hosts=['localhost'])
except TransportError as e:
    e.info()

app = Flask(__name__)
CORS(app, resources={r"/api/bert/*": {"origins": "*"}})

@app.route("/api/bert/", methods=["POST"])
def chatbot_response():
    try:
        json_data = request.get_json(force=True)

        query_string = json_data['query_string']
        top_k = json_data.get('top_k', 5)
        dataset = json_data.get('dataset', 'CovidFAQ')
        fields = json_data.get('field', ['question_answer'])
        
        # Define model parameters
        version = json_data.get('version', '1.1')
        loss_type = json_data.get('loss_type', 'Triplet')
        neg_type  = json_data.get('neg_type', 'Hard')
        query_type = json_data.get('query_type', 'USER_QUERY')

        # Get model name from model parameters
        model_name = "{}_{}_{}_{}".format(loss_type.lower(), neg_type.lower(), query_type.lower(), version)
        bert_model_path = "output" + "/" + dataset + "/models/" + model_name

        if not isDir(bert_model_path):
            response = [{"answer": "No model found with given parameters ..."}]
            return json.dumps(response)
        
        # Perform ranking
        faq_bert_ranker = FAQ_BERT_Ranker(
            es=es, index=dataset.lower(), fields=fields, top_k=top_k, bert_model_path=bert_model_path
        )

        ranked_results = faq_bert_ranker.rank_results(query_string)
      
        if ranked_results:
            return json.dumps(ranked_results)
        else:
            response = [{"answer": "No result found for the given question ..."}]
            return json.dumps(response)

    except Exception as e:
        return {"Error ": str(e)}



if __name__ == "__main__":
    app.run(debug=True, port="5000")

    
