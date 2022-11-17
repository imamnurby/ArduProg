from cherche import retrieve
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import numpy as np
import os
import json
import argparse

# DB_PATH = "../dataset_ready/db_libraries.csv"
# QUERIES_PATH = "../dataset_ready/queries_w_labels.csv"
# GROUND_TRUTH = "../"
# MODEL_NAME = "TripletLoss_uncased_iter5_sim_augmentation_roberta-2022-09-27_08-36-07"
# MODEL_PATH = f"./model/{MODEL_NAME}"
# EXTRACTED_KEYWORDS_PATH = "extracted_keywords.json"

# for name, path in zip(("DB_PATH", "DB_QUERIES", "MODEL_PATH", "EXTRACTED KEYWORDS"), (DB_PATH, QUERIES_PATH, MODEL_PATH, EXTRACTED_KEYWORDS_PATH)):
#     print(f"{name}: {os.path.exists(path)})

# prepare index
def generate_index(df):
    df_cp = df.copy()
    index_list = []
    for id_, dirname in df_cp.values:
        index_list.append(
        {
            'id': id_,
            'library': dirname.lower()
        })
    return index_list

def search_on_queries(df, model, k, with_keywords=True, model_type="deep_learning"):
    df_cp = df.copy()
    preds = []
    if model_type=="deep_learning":
        if with_keywords:
            for query_id, query, cat, truths_fam, truths_ser, keywords in df_cp.values:
                temp_preds = []
                for keyword in keywords:
                    results = model(keyword)
                    assert(len(results)==k)
                    for item in results:
                        temp_preds.append(item)    
                temp_preds_sorted = sorted(temp_preds, key=lambda d: d['similarity'], reverse=True)
                temp_preds_sorted = [x.get('id') for x in temp_preds_sorted]
                preds.append((temp_preds_sorted[:k]))
        else:
            for query_id, query, cat, truths_fam, truths_ser in df_cp.values:
                temp_preds = []
                results = model(query)
                assert(len(results)==k)
                for item in results:
                    temp_preds.append(item.get("id"))
                if len(results) < k:
                    diff = k - len(results)
                    for i in range(diff):
                        temp_preds.append(-1)
                preds.append(temp_preds)
    elif model_type=="bm25":
        if with_keywords:
            for query_id, query, cat, truths_fam, truths_ser, keywords in df_cp.values:
                temp_preds = []
                for keyword in keywords:
                    results = model(q=keyword)
                    for item in results:
                        temp_preds.append(item)
                
                    if len(results) < k:
                        diff = k - len(results)
                        for i in range(diff):
                            temp_preds.append(
                                {
                                    'id': -1,
                                    'similarity': -999
                                }
                            )
                temp_preds_sorted = sorted(temp_preds, key=lambda d: d['similarity'], reverse=True)
                temp_preds_sorted = [x.get('id') for x in temp_preds_sorted]
                preds.append(temp_preds_sorted[:k])
        else:
            for query_id, query, cat, truths_fam, truths_ser in df_cp.values:
                temp_preds = []
                results = model(q=query)
                for item in results:
                    temp_preds.append(item.get("id"))
                
                if len(results) < k:
                    diff = k - len(results)
                    for i in range(diff):
                        temp_preds.append(-1)
                preds.append(temp_preds)
    return preds

# compute ndcg
def hit_or_miss(x, k):
    preds = x.preds[:k]
    preds = [int(x) for x in preds]
    preds = np.array(preds)
    
    truths_fam = x["truths_family"] if x["truths_family"] != "null" else []
    truths_ser = x["truths_serie"] if x["truths_serie"] != "null" else []
    truths = truths_fam + truths_ser
    truths = [int(x) for x in truths]
    truths = np.array(truths)
    
    return np.in1d(preds, truths).astype(int)

def compute_dcg(relevance, alternate=True):
    if relevance is None or len(relevance) < 1:
        return 0.0
    
    rel = (relevance)
    p = len(rel)
    
    if alternate:
        # from wikipedia: "An alternative formulation of
        # DCG[5] places stronger emphasis on retrieving relevant documents"
        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, rel) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return rel[0] + (rel[1:] / log2i).sum()

def compute_ndcg(x, k):
    dcg = compute_dcg(x["hit"][:k])
    ideal_scores = [1 for j in range(k)]
    ideal_scores = np.array(ideal_scores)    
    idcg_score = compute_dcg(ideal_scores)
    return dcg/idcg_score
    
class wrappedTokenizer(RobertaTokenizer):
    def __call__(self, text_input):
        return self.tokenize(text_input)

def get_results(args):
    df = pd.read_csv(args.db_path)
    df = df[['id', 'dirname']].copy()
    index_list = generate_index(df)

    if args.model_type=="deep_learning":
        codebert = SentenceTransformer(args.model_path)

        if os.path.exists(args.retriever_pickle_path):
            os.remove(args.retriever_pickle_path)
        
        model = retrieve.Encoder(
            key = "id",
            on = "library",
            encoder = codebert.encode,
            k = 10,
            path = args.retriever_pickle_path
        )
        model = model.add(documents=index_list)
    
    elif args.model_type=="bm25":
        tokenizer = wrappedTokenizer.from_pretrained(args.tokenizer_path)
        model = retrieve.BM25Okapi(key="id", on="library", documents=index_list, k=10, tokenizer=tokenizer)

    df_queries = pd.read_csv(args.queries_path)
    df_queries.fillna("null", inplace=True)

    for column in ('truths_family', 'truths_serie'):
        df_queries[column] = df_queries[column].progress_apply(lambda x: x.split("###") if x != "null" else "null")
        df_queries[column] = df_queries[column].progress_apply(lambda x: [int(id_) for id_ in x] if x!= "null" else "null")

    if args.with_keywords:
        with open(args.keywords_path, "r") as f:
            df_queries['keywords'] = json.load(f)
        df_queries['keywords'] = df_queries['keywords'].progress_apply(lambda x: x.split("###") if x != "null" else "null")
    
    df_queries['preds'] = search_on_queries(df_queries, model, 10, args.with_keywords, args.model_type)
    df_queries['hit'] = df_queries.progress_apply(lambda x: hit_or_miss(x, args.top_k), axis=1) 

    ndcg_list = []
    for k in (1, 5, args.top_k):
        df_cp = df_queries[df_queries.truths_family != "null"].copy()
        df_cp['ndcg'] = df_cp.progress_apply(lambda x: compute_ndcg(x, k), axis=1)
        ndcg_list.append((k, df_cp.ndcg.sum()/len(df_cp)))

    df_result = pd.DataFrame.from_records(ndcg_list, columns=['top_k', 'ndcg']).set_index('top_k')
    df_result.to_csv(f"{args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--queries_path", type=str)
    parser.add_argument("--with_keywords", action="store_true")
    parser.add_argument("--keywords_path", type=str)
    parser.add_argument("--retriever_pickle_path", type=str)
    parser.add_argument("--top_k", type=int)

    args = parser.parse_args()
    get_results(args)







