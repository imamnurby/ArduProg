import streamlit as st
from backend_utils import load_db, load_retrieval_model_lexical, load_generative_model_codebert, load_hw_classifier, retrieve_libraries, prepare_input_generative_model, generate_api_usage_patterns, predict_hw_config
from config import config, classifier_class_mapping

st.title("ArduProg: From Hardware Setups to Sample Source Code Generation")

# main code
## load db, does not use st.session_state because the dataframe is fixed (not mutable inside a session)
db_metadata, db_constructor = load_db(
    config.get('db_metadata_path'), 
    config.get('db_constructor_path')
)

## load model
if 'model_retrieval' not in st.session_state:
    st.session_state.model_retrieval = load_retrieval_model_lexical(
        config.get('tokenizer_path_retrieval'),
        config.get('max_k'),
        db_metadata,
    )

if 'tokenizer_generative' not in st.session_state and 'model_generative' not in st.session_state:
    st.session_state.tokenizer_generative, st.session_state.model_generative = load_generative_model_codebert(config.get('model_path_generative'))

if 'model_classifier' not in st.session_state and 'head' not in st.session_state and 'tokenizer_classifier' not in st.session_state:
    st.session_state.model_classifier, st.session_state.head, st.session_state.tokenizer_classifier = load_hw_classifier(
        config.get('model_path_classifier'),
        config.get('classifier_head_path')
    )

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def predict(
    input_query, 
    model_retrieval, 
    model_generative, 
    tokenizer_generative, 
    db_metadata, 
    db_constructor,
    num_beams,
    num_return_sequences
    ):
    '''
    Function to retrieve relevant libraries, generate API usage patterns for each library, and predict the hardware configuration

    Params:
    input
    '''
    library_ids, library_names = retrieve_libraries(model_retrieval, input_query, db_metadata)

    input_generative_model_dict = prepare_input_generative_model(library_ids, db_constructor)

    output_list = []
    for id_ in input_generative_model_dict:
        temp_dict = {
            'id': id_,
            'library_name': None,
            'hw_config': None,
            'usage_patterns': {}
        }
        temp_dict['id'] = id_
        for input_generative_model in input_generative_model_dict.get(id_):
            api_usage_patterns = generate_api_usage_patterns(
                model_generative,
                tokenizer_generative,
                input_generative_model,
                num_beams,
                num_return_sequences
            )

            temp = input_generative_model.split("[SEP]")
            library_name = temp[0].strip()
            constructor = temp[1].strip()

            assert(constructor not in temp_dict.get('usage_patterns'))
            temp_dict['usage_patterns'][constructor] = api_usage_patterns
        
        assert(temp_dict.get('library_name')==None)
        temp_dict['library_name'] = library_name
        output_list.append(temp_dict)

    hw_configs = predict_hw_config(
        st.session_state.model_classifier,
        st.session_state.tokenizer_classifier,
        st.session_state.head,
        library_ids,
        db_metadata,
        config.get('max_length')
    )

    for output_dict, hw_config in zip(output_list, hw_configs):
        output_dict['hw_config'] = hw_config

    st.session_state.prediction = output_list


st.header("Enter a Query")
input_query = st.text_input(
    'Enter some text 👇',
    max_chars=150,
    key='input_query',
    placeholder='Input your query, then press Enter',
)

generate = st.button(
    label='Retrieve',
    key='generate',
    on_click=predict,
    args=(
        st.session_state.input_query,
        st.session_state.model_retrieval,
        st.session_state.model_generative,
        st.session_state.tokenizer_generative,
        db_metadata,
        db_constructor,
        config.get('num_beams'),
        config.get('num_return_sequences')
    )
)

if st.session_state.prediction != None:
    st.write(st.session_state.prediction)