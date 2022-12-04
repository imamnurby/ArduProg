from flask import Flask, request, jsonify
from backend_utils import initialize_all_components, make_predictions
from config import classifier_class_mapping, config
from flask_cors import CORS, cross_origin
import json


# todo: downgrade version sklearn to 1.0.2

app = Flask(__name__)
CORS(app)
components = initialize_all_components(config)
db_metadata = components[0]
db_constructor = components[1]
model_retrieval = components[2]
model_generative = components[3]
tokenizer_generative = components[4]
model_classifier = components[5]
classifier_head = components[6]
tokenizer_classifier = components[7]

def call_predict_api(
    input_query, 
    model_retrieval, 
    model_generative,  
    model_classifier, classifier_head,
    tokenizer_generative, tokenizer_classifier,
    db_metadata, db_constructor,
    config
    ):
    '''
    wrapper to the make prediction function
    '''
    predictions = make_predictions(
        input_query, 
        model_retrieval, 
        model_generative,  
        model_classifier, classifier_head,
        tokenizer_generative, tokenizer_classifier,
        db_metadata, db_constructor,
        config
    )
    return predictions

@app.route("/")
def hello_world():
    return "<p>Hello, World! This Yusuf</p>"

# @app.route('/predict/')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # user_query = request.args.get('user_query', None)
    # predictions = call_predict_api(
    #     user_query,
    #     model_retrieval,
    #     model_generative,
    #     model_classifier, classifier_head,
    #     tokenizer_generative, tokenizer_classifier,
    #     db_metadata, db_constructor,
    #     config
    # )

    # return jsonify(predictions)
    # return jsonify({
    #     'class_id': 'temp'
    # })
    # return {'class_id': user_query}

    request_data = request.get_json()
    user_query = request_data.get('user_query', None)

    if user_query != None:
        predictions = call_predict_api(
                user_query,
                model_retrieval,
                model_generative,
                model_classifier, classifier_head,
                tokenizer_generative, tokenizer_classifier,
                db_metadata, db_constructor,
                config
            )
        with open("prediction.txt", 'w') as f:
            json.dump(predictions, f)
        return {
            'predictions': predictions
        }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8111)
