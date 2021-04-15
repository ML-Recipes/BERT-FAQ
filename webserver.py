from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch, TransportError
from flask import Flask, request
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from pathlib import Path
import dialogflow
import json
import os

from faq_bert_ranker import FAQ_BERT_Ranker
from shared.utils import isDir

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

project_id =  os.environ.get('PROJECT_ID')
session_id = os.environ.get('SESSION_ID')
language_code = os.environ.get('LANGUAGE_CODE')
chatbot_credentials = os.environ.get('CHATBOT_CREDENTIALS')

try:
    es = connections.create_connection(hosts=['localhost'], http_auth=('elastic', 'elastic'))
except TransportError as e:
    e.info()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/chatbot/search/<dataset>', methods=['GET'])
@cross_origin()
def get_index_list(dataset):
    try:
        index_list = []
        count = -1
        for elem in es.cat.indices(format="json"):
            if elem['index'].startswith(dataset):
                count += 1
                index = elem['index']
                num_docs = es.count(index=index)["count"]
                index_list.append(
                    {
                        "label": index.split("_")[1],
                        "value": count,
                        "legend": str(num_docs)
                    }
                )

        index_list = sorted(index_list, key= lambda k: k['label'])

        ### TO-DO: get all questions for each index and extract top-k terms

        # Iterate over the list of indices to compute the diffence of documents
        index_list_ = []
        prev_num_docs = 0
        for index in index_list:

            label = index['label']
            value = index['value']
            num_docs = int(index['legend'])
            diff_docs = num_docs - prev_num_docs
            index_list_.append(
                    {
                        "label": label,
                        "value": value,
                        "legend": str(num_docs) + " <small>(+" + str(diff_docs) + ")</small>",
                        "topics": ["travel", "spread"] # 
                    }
                )
            prev_num_docs = num_docs

    except Exception as e:
        return {"Error ": str(e)}

    return json.dumps(index_list_)

@app.route("/api/chatbot/", methods=["POST"])
@cross_origin()
def chatbot_response():
    try:
        json_data = request.get_json(force=True)
        
        query_string = json_data['query_string']
        top_k = json_data.get('top_k', 5)
        dataset = json_data.get('dataset', 'CovidFAQ')
        index = json_data.get('index')
        fields = json_data.get('field', ['question_answer'])
        
        # Define model parameters
        version = json_data.get('version', '1.1')
        loss_type = json_data.get('loss_type', 'Triplet')
        neg_type  = json_data.get('neg_type', 'Hard')
        query_type = json_data.get('query_type', 'USER_QUERY')

        # Handle Dialogflow 
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)
        text_input = dialogflow.types.TextInput(text=query_string, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)
    
        if response.query_result.intent.display_name == 'Default Welcome Intent':

            return json.dumps([
                                    {
                                        "_type":  "dialogflow",
                                        "answer": response.query_result.fulfillment_text,
                                        "intent": response.query_result.intent.display_name,
                                        "confidence": response.query_result.intent_detection_confidence
                                    }
                                ]
            )
    
        else:
            # Get model name from model parameters
            model_name = "{}_{}_{}_{}".format(loss_type.lower(), neg_type.lower(), query_type.lower(), version)
            bert_model_path = "output" + "/" + dataset + "/models/" + model_name

            if not isDir(bert_model_path):
                response = [{"answer": "No model found with given parameters ..."}]
                return json.dumps(response)
            
            # Perform ranking
            faq_bert_ranker = FAQ_BERT_Ranker(
                es=es, index=index, fields=fields, top_k=top_k, bert_model_path=bert_model_path, search_mode='history'
            )

            ranked_results = faq_bert_ranker.rank_results(query_string)
        
            if ranked_results:
                return json.dumps(ranked_results)
            else:
                response = [
                    {
                        "_type": "error",
                        "answer": "Sorry I could not find any answer! Please ask again."
                    }
                ]
                return json.dumps(response)
        
    except Exception as e:
        return {"Error ": str(e)}



if __name__ == "__main__":
    app.run(debug=True, port="5000")

    
