DB_PATH="../dataset_ready/db_libraries.csv"
QUERIES_PATH="../dataset_ready/queries_w_labels.csv"
KEYWORDS_PATH="extracted_keywords.json"

RETRIEVER_PICKLE_PATH="temp/pickle_retriever"
MODEL_PATH="./model/"
MODEL_NAMES=("TripletLoss_uncased_iter5_sim_augmentation_roberta-2022-09-27_08-36-07" \
            "TripletLoss_uncased_iter5_sim_augmentation_codebert-2022-08-20_04-30-14")
MODEL_TYPE="deep_learning"
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    TEMP_MODEL_PATH=${MODEL_PATH}/${MODEL_NAME}
    python inference.py --model_type ${MODEL_TYPE} \
                            --model_path ${TEMP_MODEL_PATH} \
                            --db_path ${DB_PATH} \
                            --queries_path ${QUERIES_PATH} \
                            --keywords_path ${KEYWORDS_PATH} \
                            --retriever_pickle_path ${RETRIEVER_PICKLE_PATH} \
                            --top_k 10 \
                            --output_path ${MODEL_NAME}_results_keywords.csv \
                            --with_keywords
done

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    TEMP_MODEL_PATH=${MODEL_PATH}/${MODEL_NAME}
    python inference.py --model_type ${MODEL_TYPE} \
                            --model_path ${TEMP_MODEL_PATH} \
                            --db_path ${DB_PATH} \
                            --queries_path ${QUERIES_PATH} \
                            --keywords_path ${KEYWORDS_PATH} \
                            --retriever_pickle_path ${RETRIEVER_PICKLE_PATH} \
                            --top_k 10 \
                            --output_path ${MODEL_NAME}_results_no_keywords.csv
done

TOKENIZER_PATH="imamnurby/bow-tokenizer-uncased"
MODEL_TYPE="bm25"
python inference.py --model_type ${MODEL_TYPE} \
                            --tokenizer_path ${TOKENIZER_PATH} \
                            --db_path ${DB_PATH} \
                            --queries_path ${QUERIES_PATH} \
                            --keywords_path ${KEYWORDS_PATH} \
                            --top_k 10 \
                            --output_path bm25_results_keywords.csv \
                            --with_keywords 

python inference.py --model_type ${MODEL_TYPE} \
                            --tokenizer_path ${TOKENIZER_PATH} \
                            --db_path ${DB_PATH} \
                            --queries_path ${QUERIES_PATH} \
                            --keywords_path ${KEYWORDS_PATH} \
                            --top_k 10 \
                            --output_path bm25_results_no_keywords.csv