import argparse
import json
import jsonlines
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_pair_file", type=str, required=True)
    parser.add_argument("--corpus_file", type=str, required=True)
    parser.add_argument("--dev_query_file", type=str, required=True)
    
    parser.add_argument("--dev_corpus_file", type=str, required=True)
    parser.add_argument("--dev_pmi_file", type=str, required=True)
    args = parser.parse_args()
    return args

def read_target_pair(fname):
    pmi = {}
    with jsonlines.open(fname, "r") as reader:
        for r in tqdm(reader, desc="Reading the pmi file"):
            pmi[(r["head"], r["tail"])] = float(r["pmi"])
            #if len(pmi) > 10000:
            #    break
    return pmi

def read_jsonl(fname):
    data = []
    with jsonlines.open(fname, "r") as reader:
        for r in tqdm(reader, desc="Reading {}".format(fname.split("/")[-1])):
            data.append(r)
            #if len(data) > 1000:
            #    break
    return data

def get_dev_corpus(dev_query, id2docs, dev_corpus_file):
    dev_doc_ids = []
    for r in tqdm(dev_query, desc="Get dev corpus"):
        dev_doc_ids += r["candidate_doc_ids"]
    dev_doc_ids = list(set(dev_doc_ids))

    dev_corpus = []
    for doc_id in tqdm(dev_doc_ids, desc="Get dev corpus"):
        if doc_id not in id2docs:
            continue
        dev_corpus += id2docs[doc_id]
    
    with jsonlines.open(dev_corpus_file, "w") as writer:
        for d in tqdm(dev_corpus, desc="Writing dev corpus"):
            writer.write(d)
    return dev_corpus

def get_dev_pmi(dev_query, dev_corpus, pmi, dev_pmi_file):
    dev_pair = set([])
    for query in tqdm(dev_query, desc="Get dev pmi"):
        mentions = query["mentions"]
        pairs = query["target_entity_pairs"]
        for i, j in pairs:
            dev_pair.add((mentions[i][1], mentions[j][1]))
    for doc in tqdm(dev_corpus, desc="Get dev pmi"):
        entities = doc["entities"]
        pairs = doc["target_entity_pairs"]
        for i, j in pairs:
            dev_pair.add((entities[i][1], entities[j][1]))

    dev_pmi = []
    for pair in tqdm(dev_pair, desc="Get dev pmi"):
        if pair not in pmi:
            continue
        dev_pmi.append({
            "head": pair[0],
            "tail": pair[1],
            "pmi": pmi[pair]
        })
    with jsonlines.open(dev_pmi_file, "w") as writer:
        for _ in tqdm(dev_pmi, desc="Writing dev pmi"):
            writer.write(_)
    return 0

def main(args):
    pmi = read_target_pair(args.target_pair_file)
    corpus = read_jsonl(args.corpus_file)
    id2docs = {}
    for c in tqdm(corpus, desc="Processing corpus"):
        if c["id"] not in id2docs:
            id2docs[c["id"]] = []
        id2docs[c["id"]].append(c)
    dev_query = read_jsonl(args.dev_query_file)
    
    dev_corpus = get_dev_corpus(dev_query, id2docs, args.dev_corpus_file)

    _ = get_dev_pmi(dev_query, dev_corpus, pmi, args.dev_pmi_file)
        
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)

