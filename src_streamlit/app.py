from cherche import retrieve
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, AutoModel, AutoTokenizer
import pandas as pd
import streamlit as st


st.title("ArduProg: From Hardware Setups to Sample Source Code Generation")

# initialize global variable
model_path = {
    'dl_retrieval': '../models/TripletLoss_uncased_iter5_sim_guidance_distilbert-2022-09-29_22-01-04',
    'dl_generative': '../models/codebert2codebert'
    'lexical': None,
}

tokenizer_path = {
    'dl_retrieval': None,
    'lexical': 'imamnurby/bow-tokenizer-uncased'
}

db_path_features = '../assets/generate_features_mapping/lib_to_features.csv'
db_path_constructor = '../assets/generate_constructor_mapping/lib_to_constructor.csv'

# initialize state session variable
if "topk" not in st.session_state:
    st.session_state.topk = 10

if "is_lexical_loaded" not in st.session_state:
    st.session_state.is_lexical_loaded = False

if "is_dl_loaded" not in st.session_state:
    st.session_state.is_dl_loaded = False

if "is_db_loaded" not in st.session_state:
    st.session_state.is_db_loaded = False

if "default_library_option" not in st.session_state:
    st.session_state.default_library_option = (None, )

# helper function
def load_db(db_path_features, db_path_constructor):
    db_features = pd.read_csv(db_path_features)
    db_constructor = pd.read_csv(db_path_constructor)
    st.session_state.is_db_loaded = True
    return db_features, db_constructor

class wrappedTokenizer(RobertaTokenizer):
    def __call__(self, text_input):
        return self.tokenize(text_input)

def load_retrieval_model(model_path, tokenizer_path, topk, db):
    
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

    index_list = generate_index(db[['id', 'library']])
    id_to_libname = {item['id']: item['library'] for item in index_list}
    libname_to_id = {item['library']: item['id'] for item in index_list}

    tokenizer_retriever = wrappedTokenizer.from_pretrained(tokenizer_path['lexical'])
    lx_retriever = retrieve.BM25Okapi(
        key='id',
        on='library',
        documents=index_list,
        k=topk,
        tokenizer=tokenizer_retriever
    )
    st.session_state.is_lexical_loaded = True

    dl_retriever = retrieve.Encoder(
        key='id',
        on='library',
        encoder=SentenceTransformer(model_path['dl_retrieval']).encode,
        k=topk,
        path=f"../temp/dl.pkl"
    )
    dl_retriever = dl_retriever.add(documents=index_list)
    st.session_state.is_dl_loaded = True

    tokenizer_pattern_gen = AutoTokenizer.from_pretrained(model_path['dl_generative'])
    pattern_gen = AutoModel.from_pretrained(model_path['dl_generative']) 
    
    return dl_retriever, lx_retriever, id_to_libname, libname_to_id, tokenizer_pattern_gen, pattern_gen

def get_metadata_library(id_, db):
    temp_db = db[db.id==id_]
    assert(len(temp_db)==1)

    output_dict = {}
    output_dict['Library Name'] = temp_db.iloc[0]['library']
    output_dict['Sensor Type'] = temp_db.iloc[0]['cat'].capitalize()
    output_dict['Github URL'] = temp_db.iloc[0]['url']
    
    if temp_db.iloc[0].desc_ardulib != 'nan':
        output_dict['Description'] = temp_db.iloc[0].desc_ardulib
    
    elif temp_db.iloc[0].desc_repo != 'nan':
        output_dict['Description'] = temp_db.iloc[0].desc_repo

    else:
        output_dict['Description'] = "Description not found"
    
    return output_dict

# main code

## load db
db_features, db_constructor = load_db(db_path_features, db_path_constructor)

## load model
if 'dl_retriever' not in st.session_state and 'lx_retriever' not in st.session_state and 'tokenizer_pattern_gen' not in st.session_state and 'pattern_gen' not in st.session_state:
    st.session_state.dl_retriever, st.session_state.lx_retriever, st.session_state.id_to_libname, st.session_state.libname_to_id, st.session_state.tokenizer_pattern_gen, st.session_state.pattern_gen = load_retrieval_model(
                                    model_path=model_path,
                                    tokenizer_path=tokenizer_path,
                                    topk=st.session_state.topk,
                                    db=db_features
    )


## STEP1: enter query
st.header("STEP 1: Enter Query")
input_query = st.text_input(
    'Enter some text 👇',
    label_visibility='collapsed',
    max_chars=150,
    key='input_query',
    placeholder='Input your query, then press Enter'
)

## select model to perform retrieval
model_type = st.radio(
    label='Select the retrieval model type',
    options=('Deep Learning', 'BM25'),
    index=0,
    key='model_type'
)

## button to perform retrieval
generate = st.button(
    label='Retrieve',
    key='generate',
)

## logic
if st.session_state.generate==True:
    results = []
    translated_results = []
    translated_results.append('None')
    if st.session_state.model_type == 'Deep Learning':
        results = st.session_state.dl_retriever(input_query)

    elif st.session_state.model_type == 'BM25':
        results = st.session_state.lx_retriever(input_query)

    results = [item.get('id') for item in results]
    if 'retrieval_results' not in st.session_state:
        st.session_state.retrieval_results = results

    # st.subheader('Retrieval Results')
    for idx, result in enumerate(results):
        metadata_dict = get_metadata_library(result, db_features)
        translated_results.append(
            f'{idx+1} - {metadata_dict.get("Library Name")} - {metadata_dict.get("Description")}' 
        )
        # st.markdown(f'''
            
        #     **Prediction {idx+1}**
        #     - Library Name: {metadata_dict.get("Library Name")}
        #     - Description: {metadata_dict.get("Description")}
        #     - Sensory Category: {metadata_dict.get("Sensor Type")}
        #     - Github URL: {metadata_dict.get("Github URL")}
        #     ***
        # ''')

        # if idx == 2:
        #     break

    # if len(results) > 2:    
    #     with st.expander("See more predictions"):
    #         for idx, result in enumerate(results[3:]):
    #             metadata_dict = get_metadata_library(result, db_features)
    #             translated_results.append(
    #                 f'{metadata_dict.get("Library Name")} - {metadata_dict.get("Description")}' 
    #             )
    #             st.markdown(f'''
    #                 **Prediction {idx+4}**
    #                 - Library Name: {metadata_dict.get("Library Name")}
    #                 - Description: {metadata_dict.get("Description")}
    #                 - Sensory Category: {metadata_dict.get("Sensor Type")}
    #                 - Github URL: {metadata_dict.get("Github URL")}
    #                 ***
    #             ''')
    
    st.session_state.default_library_option = translated_results

# STEP2: Select a library, then predict hardware configuration and api sequennce pattern
st.header("STEP 2: Select a Library")
selected_library = st.selectbox(
    'Choose one of the predictions below (ranked by the similarity score)',
    key='selected_library',
    options=st.session_state.default_library_option,
    index=0,
)

if selected_library != 'None':
    ranking = selected_library.split(" - ")[0]
    metadata_dict = get_metadata_library(st.session_state.retrieval_results[int(ranking)], db_features)
    st.markdown(f'''
        - Library Name: {metadata_dict.get("Library Name")}
        - Description: {metadata_dict.get("Description")}
        - Category: {metadata_dict.get("Sensor Type")}
        - Github URL: {metadata_dict.get("Github URL")}
    ''')

# debug
# st.write(dict(st.session_state.items()))
# st.write(db)