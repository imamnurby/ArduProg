from cherche import retrieve
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, EncoderDecoderModel

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
    retriever: a retrieval model
    '''
    # generate index
    index_list = generate_index(db_metadata[['id', 'library']])

    # load model
    tokenizer = wrappedTokenizer.from_pretrained(tokenizer_path['lexical'])
    retriever = retrieve.BM25Okapi(
        key='id',
        on='library',
        documents=index_list,
        k=max_k,
        tokenizer=tokenizer
    )
    return retriever


def load_retrieval_model_deep_learning(model_path, max_k, db_metadata):
    '''
    Function to load a deep learning-based model

    Params:
    model_path (string): the path to the model (can be a path to either a huggingface model or local directory)
    max_k (int): the maximum number of returned sequences
    db_metadata (pandas dataframe): a dataframe containing metadata information about the library
    
    Returns:
    retriever: a retrieval model
    '''
    # generate index
    index_list = generate_index(db_metadata[['id', 'library']])

    # load model
    retriever = retrieve.Encoder(
        key='id',
        on='library',
        encoder=SentenceTransformer(model_path).encode,
        k=max_k,
        path=f"../temp/dl.pkl"
    )
    retriever = dl_retriever.add(documents=index_list)
    
    return retriever

def load_generative_model_codebert(model_path):
    '''
    Function load a generative model using codebert checkpoint

    Params: 
    model_path (string): path to the model (can be a path to either a huggingface model or local directory)
    
    Returns:
    tokenizer: a huggingface tokenizer
    model: a generative model to generate API pattern given the library name as the input
    '''
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    return tokenizer, model


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

def retrieve_library(model, model_input):
    '''
    Function to retrieve a set of relevant libraries using a model based on the input query

    Params:
    model: a model to perform retrieval
    model_input (string): an input query from the user

    Returns:
    results (list): a list of library unique id
    '''
    results = model(model_input)
    results = [item.get('id') for item in results]
    return results

def generate_api_patterns(model, tokenizer, model_input, num_beams, num_return_sequences):
    '''
    Function to generate API usage patterns

    Params:
    model: a huggingface model
    tokenizer: a huggingface tokenizer
    model_input (string): a string in the form of <library-name> [SEP] constructor
    num_beams (int): the beam width used for decoding
    num_return_sequences (int): how many API usage patterns are returned by the model

    Returns:
    results (list): a list of API usage patterns
    '''
    model_input = tokenizer(model_input, return_tensors='pt').input_ids
    model_output = model.generate(
        model_input,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences
    )
    results = tokenizer.batch_decode(
        model_output,
        skip_special_tokens=True
    )
    return results