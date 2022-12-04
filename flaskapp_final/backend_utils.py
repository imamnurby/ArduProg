from cherche import retrieve
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaModel, EncoderDecoderModel
from config import classifier_class_mapping, config
import pandas as pd
import numpy as np 
import pickle
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import spacy

# nlp = spacy.load("en_core_web_trf")
nlp = spacy.load("en_core_web_sm")



class wrappedTokenizer(RobertaTokenizer):
    def __call__(self, text_input):
        return self.tokenize(text_input)

def generate_index(db):
    db_cp = db.copy()
    index_list = []
    for id_, dirname in db_cp.values:
        index_list.append(
        {
            'id': id_,
            'library': dirname.lower()
        })
    return index_list

def load_db(db_metadata_path, db_constructor_path, db_params_path, exclusion_list_path):
    '''
    Function to load dataframe

    Params:
    db_metadata_path (string): the path to the db_metadata file
    db_constructor_path (string): the path to the db_constructor file

    Output:
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library
    db_constructor (pandas dataframe): a dataframe containing the mapping of library names to valid constructor
    '''
    db_metadata = pd.read_csv(db_metadata_path)
    db_metadata.dropna(inplace=True)
    db_constructor = pd.read_csv(db_constructor_path)
    db_constructor.dropna(inplace=True)
    db_params = pd.read_csv(db_params_path)
    db_params.dropna(inplace=True)
    with open(exclusion_list_path, 'r') as f:
        ex_list = f.read()
    ex_list = ex_list.split("\n")

    return db_metadata, db_constructor, db_params, ex_list



def load_retrieval_model_lexical(tokenizer_path, max_k, db_metadata):
    '''
    Function to load BM25 model

    Params:
    tokenizer_path (string): the path to a tokenizer (can be a path to either a huggingface model or local directory)
    max_k (int): the maximum number of returned sequences
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library
    
    Returns:
    retrieval_model: a retrieval model
    '''
    # generate index
    index_list = generate_index(db_metadata[['id', 'library']])

    # load model
    tokenizer = wrappedTokenizer.from_pretrained(tokenizer_path)
    retrieval_model = retrieve.BM25Okapi(
        key='id',
        on='library',
        documents=index_list,
        k=max_k,
        tokenizer=tokenizer
    )
    return retrieval_model


def load_retrieval_model_deep_learning(model_path, max_k, db_metadata):
    '''
    Function to load a deep learning-based model

    Params:
    model_path (string): the path to the model (can be a path to either a huggingface model or local directory)
    max_k (int): the maximum number of returned sequences
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library
    
    Returns:
    retrieval_model: a retrieval model
    '''
    # generate index
    index_list = generate_index(db_metadata[['id', 'library']])

    # load model
    retrieval_model = retrieve.Encoder(
        key='id',
        on='library',
        encoder=SentenceTransformer(model_path).encode,
        k=max_k,
        path=f"../temp/dl.pkl"
    )
    retrieval_model = dl_retriever.add(documents=index_list)
    
    return retrieval_model

def load_generative_model_codebert(model_path):
    '''
    Function load a generative model using codebert checkpoint

    Params: 
    model_path (string): path to the model (can be a path to either a huggingface model or local directory)
    
    Returns:
    tokenizer: a huggingface tokenizer
    generative_model: a generative model to generate API pattern given the library name as the input
    '''
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    generative_model = EncoderDecoderModel.from_pretrained(model_path)
    return tokenizer, generative_model


def get_metadata_library(predictions, db_metadata):
    '''
    Function to get the metadata of a library using the library unique id

    Params:
    predictions (list): a list of dictionary containing the prediction details
    db_metadata: a dataframe containing metadata information about the library

    Returns:
    metadata_dict (dict): a dictionary where the key is the metadata type and the value is the metadata value
    '''
    predictions_cp = predictions.copy()
    for prediction_dict in predictions_cp:
        temp_db = db_metadata[db_metadata.id==prediction_dict.get('id')]
        assert(len(temp_db)==1)

        prediction_dict['Sensor Type'] = temp_db.iloc[0]['cat'].capitalize()
        prediction_dict['Github URL'] = temp_db.iloc[0]['url']
        
        # prefer the description from the arduino library list, if not found use the repo description
        if temp_db.iloc[0].desc_ardulib != 'nan':
            prediction_dict['Description'] = temp_db.iloc[0].desc_ardulib
        
        elif temp_db.iloc[0].desc_repo != 'nan':
            prediction_dict['Description'] = temp_db.iloc[0].desc_repo

        else:
            prediction_dict['Description'] = "Description not found"
    return predictions_cp

def id_to_libname(id_, db_metadata):
    '''
    Function to convert a library id to its library name

    Params:
    id_ (int): a unique library id
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library

    Returns:
    library_name (string): the library name that corresponds to the input id
    '''
    temp_db = db_metadata[db_metadata.id==id_]
    assert(len(temp_db)==1)
    library_name = temp_db.iloc[0].library
    return library_name


def retrieve_libraries(retrieval_model, model_input, db_metadata):
    '''
    Function to retrieve a set of relevant libraries using a model based on the input query

    Params:
    retrieval_model: a model to perform retrieval
    model_input (string): an input query from the user

    Returns:
    library_ids (list): a list of library unique ids
    library_names (list): a list of library names
    '''
    results = retrieval_model(model_input)
    library_ids = [item.get('id') for item in results]
    scores = [item.get('similarity') for item in results]
    library_names = [id_to_libname(item, db_metadata) for item in library_ids]
    return library_ids, library_names, scores

def prepare_input_generative_model(library_ids, db_constructor):
    '''
    Function to prepare the input of the model to generate API usage patterns

    Params:
    library_ids (list): a list of library ids
    db_constructor (pandas dataframe): a dataframe containing the mapping of library names to valid constructor

    Returns:
    output_dict (dictionary): a dictionary where the key is library id and the value is a list of valid inputs
    '''
    output_dict = {}
    for id_ in library_ids:
        temp_db = db_constructor[db_constructor.id==id_]
        output_dict[id_] = []
        for id__, library_name, methods, constructor in temp_db.values:
            output_dict[id_].append(
                f'{library_name} [SEP] {constructor}'
            )
    return output_dict

def generate_api_usage_patterns(generative_model, tokenizer, model_input, num_beams, num_return_sequences, max_length):
    '''
    Function to generate API usage patterns

    Params:
    generative_model: a huggingface model
    tokenizer: a huggingface tokenizer
    model_input (string): a string in the form of <library-name> [SEP] constructor
    num_beams (int): the beam width used for decoding
    num_return_sequences (int): how many API usage patterns are returned by the model

    Returns:
    api_usage_patterns (list): a list of API usage patterns
    '''
    model_input = tokenizer(model_input, return_tensors='pt').input_ids
    model_output = generative_model.generate(
        model_input,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences, 
        early_stopping=True,
        max_length=max_length
    )
    api_usage_patterns = tokenizer.batch_decode(
        model_output,
        skip_special_tokens=True
    )
    return api_usage_patterns

def add_params(api_usage_patterns, db_params, library_id):
    patterns_cp = api_usage_patterns.copy()
    valid = True
    processed_sequences = []
    for sequence in patterns_cp:
        sequence_list = sequence.split()

        if len(sequence_list) < 2:
            continue

        temp_list = []
        ref_obj = ''
        for api in sequence_list:
            temp_db = db_params[(db_params.id==library_id) & (db_params.methods==api.split(".")[-1])]
            
            if ref_obj == '':
                ref_obj = api.split(".")[0]
            
            if len(temp_db) > 0 and ref_obj == api.split(".")[0]:
                param = temp_db.iloc[0].params
                new_api = api + param
                temp_list.append(new_api)
            else:
                valid = False
                ref_obj = ''
                break
        
        if valid:
            processed_sequences.append("[API-SEP]".join(temp_list))
        else:
            valid = True
    return processed_sequences


def generate_api_usage_patterns_batch(generative_model, tokenizer, library_ids, db_constructor, db_params, num_beams, num_return_sequences, max_length):
    '''
    Function to generate API usage patterns in batch

    Params:
    generative_model: a huggingface model
    tokenizer: a huggingface tokenizer
    library_ids (list): a list of libary ids
    db_constructor (pandas dataframe):  a dataframe containing the mapping of library names to valid constructor
    num_beams (int): the beam width used for decoding
    num_return_sequences (int): how many API usage patterns are returned by the model

    Returns:
    predictions (list): a list of dictionary containing the api usage patterns, library name, and id
    '''
    input_generative_model_dict = prepare_input_generative_model(library_ids, db_constructor)

    predictions = []
    for id_ in input_generative_model_dict:
        temp_dict = {
            'id': id_,
            'library_name': None,
            'hw_config': None,
            'usage_patterns': {}
        }
        for input_generative_model in input_generative_model_dict.get(id_):
            api_usage_patterns = generate_api_usage_patterns(
                generative_model,
                tokenizer,
                input_generative_model,
                num_beams,
                num_return_sequences,
                max_length
            )

            temp = input_generative_model.split("[SEP]")
            library_name = temp[0].strip()
            constructor = temp[1].strip()

            assert(constructor not in temp_dict.get('usage_patterns'))
            api_usage_patterns = add_params(api_usage_patterns, db_params, id_)
            temp_dict['usage_patterns'][constructor] = api_usage_patterns
        
        assert(temp_dict.get('library_name')==None)
        temp_dict['library_name'] = library_name
        predictions.append(temp_dict)
    return predictions

# def generate_api_usage_patterns(generative_model, tokenizer, model_inputs, num_beams, num_return_sequences):
#     '''
#     Function to generate API usage patterns

#     Params:
#     generative_model: a huggingface model
#     tokenizer: a huggingface tokenizer
#     model_inputs (list): a list of <library-name> [SEP] <constructor>
#     num_beams (int): the beam width used for decoding
#     num_return_sequences (int): how many API usage patterns are returned by the model

#     Returns:
#     api_usage_patterns (list): a list of API usage patterns
#     '''
#     model_inputs = tokenizer(
#         model_inputs, 
#         max_length=max_length,
#         padding='max_length',
#         return_tensors='pt',
#         truncation=True)
    
#     model_output = generative_model.generate(
#         **model_inputs,
#         num_beams=num_beams,
#         num_return_sequences=num_return_sequences
#     )
#     api_usage_patterns = tokenizer.batch_decode(
#         model_output,
#         skip_special_tokens=True
#     )

#     api_usage_patterns = [api_usage_patterns[i:i+num_return_sequences] for i in range(0, len(api_usage_patterns), num_return_sequences)] 
#     return api_usage_patterns

def prepare_input_classification_model(id_, db_metadata):
    '''
    Function to get a feature for a classification model using library id

    Params:
    id_ (int): a unique library id
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library

    Returns:
    feature (string): a feature used for the classification model input 
    '''
    temp_db = db_metadata[db_metadata.id == id_]
    assert(len(temp_db)==1)
    feature = temp_db.iloc[0].features
    return feature

def load_hw_classifier(model_path_classifier, model_path_classifier_head):
    '''
    Function to load a classifier model and classifier head

    Params:
    model_path_classifier (string): path to the classifier checkpoint (can be either huggingface path or local directory)
    model_path_classifier_head (string): path to the classifier head checkpoint (should be a local directory)

    Returns:
    classifier_model: a huggingface model
    classifier_head: a classifier model (can be either svm or rf)
    tokenizer: a huggingface tokenizer
    '''
    tokenizer = RobertaTokenizer.from_pretrained(model_path_classifier)
    classifier_model = RobertaModel.from_pretrained(model_path_classifier)
    with open(model_path_classifier_head, 'rb') as f:
        classifier_head = pickle.load(f)
    return classifier_model, classifier_head, tokenizer

def predict_hw_config(classifier_model, classifier_tokenizer, classifier_head, library_ids, db_metadata, max_length):
    '''
    Function to predict hardware configs

    Params:
    classifier_model: a huggingface model to convert a feature to a feature vector
    classifier_tokenizer: a huggingface tokenizer
    classifier_head: a classifier head
    library_ids (list): a list of library ids
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library
    max_length (int): max length of the tokenizer output

    Returns:
    prediction (list): a list of prediction
    '''
    
    features = [prepare_input_classification_model(id_, db_metadata) for id_ in library_ids]
    tokenized_features = classifier_tokenizer(
            features,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
    with torch.no_grad():
        embedding_features = classifier_model(**tokenized_features).pooler_output.numpy()
    prediction = classifier_head.predict_proba(embedding_features).tolist()
    prediction = np.argmax(prediction, axis=1).tolist()
    prediction = [classifier_class_mapping.get(idx) for idx in prediction]
    return prediction


def initialize_all_components(config):
    '''
    Function to initialize all components of ArduProg

    Params:
    config (dict): a dictionary containing the configuration to initialize all components

    Returns:
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library
    db_constructor (pandas dataframe): a dataframe containing the mapping of library names to valid constructor
    model_retrieval, model_generative : a huggingface model
    tokenizer_generative, tokenizer_classifier: a huggingface tokenizer
    model_classifier: a huggingface model
    classifier_head: a random forest model
    '''
    # load db
    db_metadata, db_constructor, db_params, ex_list = load_db(
        config.get('db_metadata_path'), 
        config.get('db_constructor_path'),
        config.get('db_params_path'),
        config.get('exclusion_list_path')
    )

    # load model
    model_retrieval = load_retrieval_model_lexical(
        config.get('tokenizer_path_retrieval'),
        config.get('max_k'),
        db_metadata,
    )

    tokenizer_generative, model_generative = load_generative_model_codebert(config.get('model_path_generative'))

    model_classifier, classifier_head, tokenizer_classifier = load_hw_classifier(
        config.get('model_path_classifier'),
        config.get('classifier_head_path')
    )

    return db_metadata, db_constructor, db_params, ex_list, model_retrieval, model_generative, tokenizer_generative, model_classifier, classifier_head, tokenizer_classifier

def make_predictions(input_query, 
    model_retrieval, 
    model_generative,  
    model_classifier, classifier_head,
    tokenizer_generative, tokenizer_classifier,
    db_metadata, db_constructor, db_params, ex_list,
    config):
    '''
    Function to retrieve relevant libraries, generate API usage patterns, and predict the hw configs

    Params:
    input_query (string): a query from the user
    model_retrieval, model_generative, model_classifier: a huggingface model
    classifier_head: a random forest classifier
    toeknizer_generative, tokenizer_classifier: a hugggingface tokenizer,
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library
    db_constructor (pandas dataframe): a dataframe containing the mapping of library names to valid constructor
    config (dict): a dictionary containing the configuration to initialize all components
    
    Returns:
    predictions (list): a list of dictionary containing the prediction details
    '''
    print("retrieve libraries")
    queries = extract_keywords(input_query.lower(), ex_list)

    temp_list = []
    for query in queries:
        temp_library_ids, temp_library_names, temp_scores = retrieve_libraries(model_retrieval, query, db_metadata)

        if len(temp_library_ids) > 0:
            for id_, name, score in zip(temp_library_ids, temp_library_names, temp_scores):
                temp_list.append((id_, name, score))
    
    library_ids = []
    library_names = []
    if len(temp_list) > 0:
        sorted_list = sorted(temp_list, key=lambda tup: tup[2], reverse=True)
        sorted_list = sorted_list[:config.get('max_k')]
        for item in sorted_list:
            library_ids.append(item[0])
            library_names.append(item[1])

    if len(library_ids) == 0:
        print("null libraries")
        return []

    print("generate usage patterns")
    predictions = generate_api_usage_patterns_batch(
        model_generative,
        tokenizer_generative,
        library_ids,
        db_constructor,
        db_params,
        config.get('num_beams'),
        config.get('num_return_sequences'),
        config.get('max_length_generate')
    )
    
    print("generate hw configs")
    hw_configs = predict_hw_config(
        model_classifier,
        tokenizer_classifier,
        classifier_head,
        library_ids,
        db_metadata,
        config.get('max_length')
    )

    for output_dict, hw_config in zip(predictions, hw_configs):
        output_dict['hw_config'] = hw_config
    
    print("finished the predictions")
    predictions = get_metadata_library(predictions, db_metadata)

    return predictions

def extract_series(x):
    '''
    Helper function to extract i/o hw name

    Params:
    x (string): an input string
    
    Returns:
    series (list): a list i/o hw name
    '''
    name = x.replace("-", " ").replace("_", " ")
    name = name.split()
    series = []
    for token in name:
        if token.isalnum() and not(token.isalpha()) and not(token.isdigit()):
            series.append(token)
    if len(series) > 0:
        return series
    else:
        return [x]

def extract_keywords(query, ex_list):
    '''
    Function extract relevant keywords from a given query

    Params:
    query (string): a query from the user
    ex_list (list): a list of common words
    
    Returns:
    filtered_keyword_candidates (list): a list of keywords
    '''
    doc = nlp(query)
    keyword_candidates = []

    # extract keywords
    for chunk in doc.noun_chunks:
        temp_list = []

        for token in chunk:
            if token.text not in ex_list and token.pos_ not in ("DET", "PRON", "CCONJ", "NUM"):
                temp_list.append(token.text)

        if len(temp_list) > 0:
            keyword_candidates.append(" ".join(temp_list))

    filtered_keyword_candidates = []
    for keyword in keyword_candidates:
        temp_candidates = extract_series(keyword)

        for keyword in temp_candidates:

            if len(keyword.split()) > 1:
                doc = nlp(keyword)
                for chunk in doc.noun_chunks:
                    filtered_keyword_candidates.append(chunk.root.text)
            else:
                filtered_keyword_candidates.append(keyword)

    if len(filtered_keyword_candidates) == 0:
        filtered_keyword_candidates.append(query)
    
    return filtered_keyword_candidates
