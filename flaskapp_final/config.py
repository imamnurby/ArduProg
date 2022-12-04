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
    'db_params_path' : '../assets/generate_constructor_mapping/lib_to_constructor_w_params_v2.csv',
    'exclusion_list_path': '../assets/common_words.txt',
    # retrieval_model_setting
    'max_k': 10,

    # generative_model_setting
    'num_beams': 2,
    'num_return_sequences': 2,
    'max_length_generate': 100,

    # hw_classifier setting
    'max_length': 100
}

classifier_class_mapping = {
    0: {'protocol': 'UART',
        'pin_connection_from_hw_to_arduino': {
            'arduino_mega': [('RX-->19, TX-->18')],
            'arduino_uno': [('RX-->0, TX-->1')]
        }
    },
    1: {'protocol': 'SPI',
        'pin_connection_from_hw_to_arduino': {
            'arduino_mega': [('SCK-->52, MOSI-->51, MISO-->50, CS-->53')],
            'arduino_uno': [('SCK-->13, MOSI-->11, MISO-->12, CS-->10')]
        }
    },
    2: {'protocol': 'I2C',
        'pin_connection_from_hw_to_arduino': {
            'arduino_mega': [('SDA-->20, SCL-->21')],
            'arduino_uno': [('SDA-->A4, SCL-->A5')],
        }
    },
    3: {'protocol': 'Explicit declaration',
        'pin_connection_from_hw_to_arduino': {
            'arduino_mega': [('DATA-->any digital or analog pins')],
            'arduino_uno': [('DATA-->any digital or analog pins')],
        }
    }
}

