from cherche import retrieve
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer
import pandas as pd
import streamlit as st

st.title("ArduProg: From Hardware Setups to Sample Source Code Generation")

# initialize global variable
model_path = {
    'dl': '../models/TripletLoss_uncased_iter5_sim_guidance_distilbert-2022-09-29_22-01-04',
    'lexical': None,
}

tokenizer_path = {
    'dl': None,
    'lexical': 'imamnurby/bow-tokenizer-uncased'
}

db_path = '../assets/df_merge_4.csv'

# initialize state session variable
if "topk" not in st.session_state:
    st.session_state.topk = 10

if "is_lexical_loaded" not in st.session_state:
    st.session_state.is_lexical_loaded = False

if "is_dl_loaded" not in st.session_state:
    st.session_state.is_dl_loaded = False

if "is_db_loaded" not in st.session_state:
    st.session_state.is_db_loaded = False

# helper function
def load_db(db_path):
    db = pd.read_csv(db_path)
    st.session_state.is_db_loaded = True
    return db

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

    index_list = generate_index(db[['id', 'dirname']])
    id_to_libname = {item['id']: item['library'] for item in index_list}
    libname_to_id = {item['library']: item['id'] for item in index_list}

    tokenizer = wrappedTokenizer.from_pretrained(tokenizer_path['lexical'])
    lx_retriever = retrieve.BM25Okapi(
        key='id',
        on='library',
        documents=index_list,
        k=topk,
        tokenizer=tokenizer
    )
    st.session_state.is_lexical_loaded = True

    dl_retriever = retrieve.Encoder(
        key='id',
        on='library',
        encoder=SentenceTransformer(model_path['dl']).encode,
        k=topk,
        path=f"../temp/dl.pkl"
    )
    dl_retriever = dl_retriever.add(documents=index_list)
    st.session_state.is_dl_loaded = True
    
    return dl_retriever, lx_retriever, id_to_libname, libname_to_id
        

# main code

## load db
db = load_db(db_path)

## load model
if 'dl_retriever' not in st.session_state and 'lx_retriever' not in st.session_state and 'id_to_libname' not in st.session_state and 'libname_to_id' not in st.session_state:
    st.session_state.dl_retriever, st.session_state.lx_retriever, st.session_state.id_to_libname, st.session_state.libname_to_id = load_retrieval_model(
                                    model_path=model_path,
                                    tokenizer_path=tokenizer_path,
                                    topk=st.session_state.topk,
                                    db=db
    )


## enter query
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
    label='Generate',
    key='generate',
)

if st.session_state.generate==True:
    results = []
    if st.session_state.model_type=="Deep Learning":
        results = st.session_state.dl_retriever(input_query)

    elif st.session_state.model_type=="BM25":
        results = st.session_state.lx_retriever(input_query)

    results = [item.get("id") for item in results]


    translated_results = [st.session_state.id_to_libname[id_] for id_ in results]

    st.write(translated_results)
    st.write(results)


# debug
# st.write(dict(st.session_state.items()))
st.write(db)