from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch, TransportError
from flask import Flask, request
from flask_cors import CORS
import json

from faq_bert_ranker import FAQ_BERT_Ranker

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

        if dataset not in {'CovidFAQ', 'FAQIR', 'StackFAQ'}:
            return json.dumps({"Error":  "{} dataset not exists!".format(dataset)})
        
        if loss_type not in {'Softmax', 'Triplet'}:
            return json.dumps({"Error":  "{} loss_type not exists!".format(loss_type)})

        if neg_type not in {'Simple', 'Hard'}:
            return json.dumps({"Error":  "{} neg_type not exists!".format(neg_type)})
        
        if query_type not in {'FAQ', 'USER_QUERY'}:
            return json.dumps({"Error":  "{} neg_type not exists!".format(query_type)})

        # Get model name from model parameters
        model_name = "{}_{}_{}_{}".format(loss_type.lower(), neg_type.lower(), query_type.lower(), version)
        bert_model_path = "output" + "/" + dataset + "/models/" + model_name
  
        # Perform ranking
        faq_bert_ranker = FAQ_BERT_Ranker(
            es=es, index=dataset.lower(), fields=fields, top_k=top_k, bert_model_path=bert_model_path
        )

        ranked_results = faq_bert_ranker.rank_results(query_string)
        response = json.dumps(ranked_results)
        
        return response

    except Exception as e:
        return {"Error ": str(e)}



if __name__ == "__main__":
    app.run(debug=True, port="5000")

    
