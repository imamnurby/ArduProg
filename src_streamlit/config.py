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

# classifier_class_mapping = {
#     0: 'UART',
#     1: 'SPI',
#     2: 'I2C',
#     3: 'Explicit declaration'
# }

classifier_class_mapping = {
    0: {'protocol': 'UART',
        'pin_connection_from_hw_to_arduino': {
            'arduino_mega': {
                'RX-TX': ['0-1', '19-18', '17-16', '15-14'], 
            },
            'arduino_uno': {
                'RX-TX': ['0-1']
            },
        }
    },
    1: {'protocol': 'SPI',
        'pin_connection_from_hw_to_arduino': {
            'arduino_mega': {
                'SCK-MOSI-MISO-CS': ['52-51-50-53'], 
            },
            'arduino_uno': {
                'SCK-MOSI-MISO-CS': ['13-11-12-10']
            },
        },
    },
    2: {'protocol': 'I2C',
        'pin_connection_from_hw_to_arduino': {
            'arduino_mega': {
                'SDA-SCL': ['20-21'], 
            },
            'arduino_uno': {
                'SDA-SCL': ['A4-A5']
            },
        },
    },
    3: {'protocol': 'Explicit declaration',
        'pin_mapping_hardware_to_arduino': 'any digital or analog pins'
    }
}

