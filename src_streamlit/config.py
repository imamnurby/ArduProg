config = {
    # tokenizer
    'tokenizer_path_retrieval': 'imamnurby/bow-tokenizer-uncased',

    # model_path
    'model_path_retrieval': '../models/TripletLoss_uncased_iter5_sim_guidance_distilbert-2022-09-29_22-01-04',
    'model_path_generative': '../models/codebert2codebert',

    # db path
    'db_metadata_path': '../assets/generate_features_mapping/lib_to_features.csv',
    'db_constructor_path': '../assets/generate_constructor_mapping/lib_to_constructor.csv',

    # retrieval_model_setting
    'max_k': 10,

    # generative_model_setting
    'num_beams': 5,
    'num_return_sequences': 3
}