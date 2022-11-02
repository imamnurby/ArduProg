from cherche import retrieve
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaModel, EncoderDecoderModel
from config import classifier_class_mapping
import pandas as pd
import numpy as np 
import pickle
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

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

def load_db(db_metadata_path, db_constructor_path):
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
    db_constructor = pd.read_csv(db_constructor_path)
    return db_metadata, db_constructor



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


def get_metadata_library(id_, db_metadata):
    '''
    Function to get the metadata of a library using the library unique id

    Params:
    id_ (int): a library unique id
    db_metadata: a dataframe containing metadata information about the library

    Returns:
    metadata_dict (dict): a dictionary where the key is the metadata type and the value is the metadata value
    '''
    temp_db = db[db.id==id_]
    assert(len(temp_db)==1)

    metadata_dict = {}
    metadata_dict['Library Name'] = temp_db.iloc[0]['library']
    metadata_dict['Sensor Type'] = temp_db.iloc[0]['cat'].capitalize()
    metadata_dict['Github URL'] = temp_db.iloc[0]['url']
    
    # prefer the description from the arduino library list, if not found use the repo description
    if temp_db.iloc[0].desc_ardulib != 'nan':
        metadata_dict['Description'] = temp_db.iloc[0].desc_ardulib
    
    elif temp_db.iloc[0].desc_repo != 'nan':
        metadata_dict['Description'] = temp_db.iloc[0].desc_repo

    else:
        metadata_dict['Description'] = "Description not found"
    
    return metadata_dict

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
    library_names = [id_to_libname(item, db_metadata) for item in library_ids]
    return library_ids, library_names

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

def generate_api_usage_patterns(generative_model, tokenizer, model_input, num_beams, num_return_sequences):
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
        num_return_sequences=num_return_sequences
    )
    api_usage_patterns = tokenizer.batch_decode(
        model_output,
        skip_special_tokens=True
    )
    return api_usage_patterns

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

class hw_classifier:
    def __init__(self, embedding_model_path, classifier_path, class_mapping):
        self.tokenizer = RobertaTokenizer.from_pretrained(embedding_model_path)
        self.embedding_model = RobertaModel.from_pretrained(embedding_model_path)
        with open(classifier_path, 'rb') as f:
            self.classifier_head = pickle.load(f)
        self.class_mapping = class_mapping
    
    def predict_config(self, features, max_length):
        tokenized_features = self.tokenizer(
            features,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        with torch.no_grad():
            embedding_features = self.embedding_model(**tokenized_features).pooler_output.numpy()
        prediction = self.classifier_head.predict_proba(embedding_features).tolist()
        # prediction = np.argmax(prediction, axis=1).tolist()
        # prediction = [self.class_mapping.get(idx) for idx in prediction]
        return prediction

# def predict_hw_config(model, library_ids, db_metadata, max_length):
#     '''
#     Function to predict hardware configuration

#     Params:
#     model (hw_classifier): a classifier
#     library_ids (list): a list of library ids

#     Returns:
#     hw_configs (list): a list of hardware configurations
#     '''
#     features = [prepare_input_classification_model(id_, db_metadata) for id_ in library_ids]
#     hw_configs = model.predict_config(features, max_length)
#     return hw_configs

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


