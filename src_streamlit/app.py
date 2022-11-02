import streamlit as st
from backend_utils import initialize_all_components, retrieve_libraries, generate_api_usage_patterns_batch, predict_hw_config
from config import classifier_class_mapping, config

st.title("ArduProg: From Hardware Setups to Sample Source Code Generation")

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if st.session_state.initialized == False:
    for key in ('model_retrieval',
    'model_generative',
    'tokenizer_generative',
    'model_classifier',
    'classifier_head',
    'tokenizer_classifier',
    'prediction',
    'db_metadata',
    'db_constructor'):
        if key not in st.session_state:
            st.session_state[key] = None
    
    components = initialize_all_components(config)
    st.session_state.db_metadata = components[0]
    st.session_state.db_constructor = components[1]
    st.session_state.model_retrieval = components[2]
    st.session_state.model_generative = components[3]
    st.session_state.tokenizer_generative = components[4]
    st.session_state.model_classifier = components[5]
    st.session_state.classifier_head = components[6]
    st.session_state.tokenizer_classifier = components[7]
    st.session_state.initialized = True

def predict(
    input_query, 
    model_retrieval, 
    model_generative,  
    model_classifier, classifier_head,
    tokenizer_generative, tokenizer_classifier,
    db_metadata, db_constructor,
    config
    ):
    '''
    Function to retrieve relevant libraries, generate API usage patterns for each library, and predict the hardware configuration

    Params:
    input
    '''
    library_ids, library_names = retrieve_libraries(model_retrieval, input_query, db_metadata)

    predictions = generate_api_usage_patterns_batch(
        model_generative,
        tokenizer_generative,
        library_ids,
        db_constructor,
        config.get('num_beams'),
        config.get('num_return_sequences')
    )
    
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

    st.session_state.prediction = predictions


st.header("Enter a Query")
st.write(st.session_state)
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
        st.session_state.model_classifier, st.session_state.classifier_head,
        st.session_state.tokenizer_generative, st.session_state.tokenizer_classifier,
        st.session_state.db_metadata, st.session_state.db_constructor,
        config
    )
)

if st.session_state.prediction != None:
    st.write(st.session_state.prediction)