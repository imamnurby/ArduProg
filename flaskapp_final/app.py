from flask import Flask, request, jsonify, render_template
from backend_utils import initialize_all_components, make_predictions
from config import classifier_class_mapping, config
import json

# todo: downgrade version sklearn to 1.0.2

app = Flask(__name__)
components = initialize_all_components(config)
db_metadata = components[0]
db_constructor = components[1]
db_params = components[2]
ex_list = components[3]
model_retrieval = components[4]
model_generative = components[5]
tokenizer_generative = components[6]
model_classifier = components[7]
classifier_head = components[8]
tokenizer_classifier = components[9]

def call_predict_api(
    input_query, 
    model_retrieval, 
    model_generative,  
    model_classifier, classifier_head,
    tokenizer_generative, tokenizer_classifier,
    db_metadata, db_constructor, db_params, ex_list,
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
        db_metadata, db_constructor, db_params, ex_list,
        config
    )
    return predictions

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    user_query = request.args.get("user_query")
    print(f"user_query: {user_query}")
    
    predictions = []
    if user_query != None:
        print("predicting")
        predictions = call_predict_api(
                user_query,
                model_retrieval,
                model_generative,
                model_classifier, classifier_head,
                tokenizer_generative, tokenizer_classifier,
                db_metadata, db_constructor, db_params, ex_list,
                config
            )
        
    return jsonify({
        'predictions': predictions
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8111)
