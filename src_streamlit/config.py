config = {
    # tokenizer
    'tokenizer_path_retrieval': 'imamnurby/bow-tokenizer-uncased',

    # model_path
    'model_path_retrieval': '../models/retrieval_distillbert',
    'model_path_generative': '../models/generative_codebert2codebert',
    'model_path_classifier': '../models/classifier_codebert_11nov_latest',
    'classifier_head_path': '../models/classifier_head_rf_11nov.pkl',

    # db path
    'db_metadata_path': '../assets/generate_features_mapping/lib_to_features.csv',
    'db_constructor_path': '../assets/generate_constructor_mapping/lib_to_constructor.csv',

    # retrieval_model_setting
    'max_k': 10,

    # generative_model_setting
    'num_beams': 5,
    'num_return_sequences': 3,

    # hw_classifier setting
    'max_length': 512
}

classifier_class_mapping = {
    0: 'UART',
    1: 'SPI',
    2: 'I2C',
    3: 'Explicit declaration'
}