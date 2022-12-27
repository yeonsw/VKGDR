import os

import argparse
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever 
import json
import jsonlines
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_inpf", type=str, required=True)
    parser.add_argument("--q_inpf", type=str, required=True)
    parser.add_argument("--outf", type=str, required=True)
    args = parser.parse_args()
    return args

def preprocess_docs(docs):
    documents = []
    for doc in tqdm(docs, desc="Preprocessing docs"):
        documents.append({ \
            "content": doc["text"], "meta": {"doc_id": doc["doc_id"]} \
        })
    return documents

def read_docs(inpf):
    docs = []
    with jsonlines.open(inpf, "r") as reader:
        for r in tqdm(reader, desc="Reading docs"):
            docs.append({
                "doc_id": r["id"],
                "text": "{}\n{}".format(r["title"], r["text"])
            })
    return docs

def read_queries(inpf):
    queries = []
    with open(inpf, "r") as f:
        data = json.load(f)
        for d in data:
            queries.append({
                "qid": d["QUESTION_ID"],
                "question": "{}\n{}".format(d["QUESTION_TITLE"], d["QUESTION_TEXT"]),
                "gt_doc_id": d["DOCUMENT"]
            })
    return queries

def main(args):
    queries = read_queries(args.q_inpf)
    docs = read_docs(args.doc_inpf)
    preprocessed_docs = preprocess_docs(docs)

    document_store = ElasticsearchDocumentStore()
    
    print("Storing documents...")
    s_time = time.time()
    document_store.write_documents(preprocessed_docs)
    e_time = time.time()
    print("Running time: {:.04f} sec".format(e_time - s_time))
    print("Creating an Elastic Retriever instance...")
    retriever = BM25Retriever(document_store=document_store)
    
    result = []
    for q in tqdm(queries, desc="Retrieval"):
        res = retriever.retrieve( \
            query=q["question"], top_k=50 \
        )
        res = [ \
            {"doc_id": r.meta["doc_id"], "bm25_score": r.score} \
                for r in res \
        ]
        res = sorted(res, key=lambda x: x["bm25_score"], reverse=True)
        result.append({
            "qid": q["qid"],
            "bm25_results": res,
            "gt_doc_id": q["gt_doc_id"]
        })
    os.makedirs(os.path.dirname(args.outf), exist_ok=True)
    with jsonlines.open(args.outf, "w") as writer:
        for r in tqdm(result, desc="Writing results"):
            writer.write(r)
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
