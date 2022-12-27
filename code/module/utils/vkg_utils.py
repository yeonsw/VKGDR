import os

import json
import jsonlines
import numpy as np
from tqdm import tqdm

def read_vkg(vkg_path, n_file="n_batches.json", vec_file="rel_vec.npy", ht_file="ht.jsonl"):
    with open(os.path.join(vkg_path, n_file), "r") as f:
        n = json.load(f)
    
    with open(os.path.join(vkg_path, vec_file), "rb") as f:
        rel_vecs = []
        for i in tqdm(range(n), desc="Reading the rel_vec file"):
            vecs = np.load(f).astype(np.float16)
            rel_vecs += [v for v in vecs]

    with jsonlines.open(os.path.join(vkg_path, ht_file), "r") as reader:
        hts = []
        for r in tqdm(reader, desc="Reading the ht file"):
            hts.append(r)
            if len(hts) == len(rel_vecs):
                break
    
    docid2vecs = {}
    for ht, rel in tqdm(zip(hts, rel_vecs), desc="Constructing vkg"):
        head = ht["head"]
        tail = ht["tail"]
        doc_id = ht["doc_id"]
        if doc_id not in docid2vecs:
            docid2vecs[doc_id] = {}
            docid2vecs[doc_id]["vecs"] = []
            docid2vecs[doc_id]["head_tails"] = []
        docid2vecs[doc_id]["vecs"].append(rel)
        docid2vecs[doc_id]["head_tails"].append((head, tail))
    
    rel_vecs = None
    for doc_id in tqdm(docid2vecs, desc="Postprocessing (docid2vecs)"):
        docid2vecs[doc_id]["vecs"] = np.stack(docid2vecs[doc_id]["vecs"], axis=0)

    return docid2vecs

def transform_vkg(vkg_path, n_file="n_batches.json", vec_file="rel_vec.npy", ht_file="ht.jsonl"):
    with open(os.path.join(vkg_path, n_file), "r") as f:
        n = json.load(f)
    
    with open(os.path.join(vkg_path, vec_file), "rb") as f:
        rel_vecs = []
        for i in tqdm(range(n), desc="Reading the rel_vec file"):
            vecs = np.load(f).astype(np.float16)
            rel_vecs += [v for v in vecs]
    
    dim = rel_vecs[0].shape[0]

    with jsonlines.open(os.path.join(vkg_path, ht_file), "r") as reader:
        hts = []
        for r in tqdm(reader, desc="Reading the ht file"):
            hts.append(r)
    
    docid2vecs = {}
    for ht, rel in tqdm(zip(hts, rel_vecs), desc="Constructing vkg"):
        head = ht["head"]
        tail = ht["tail"]
        doc_id = ht["doc_id"]
        if doc_id not in docid2vecs:
            docid2vecs[doc_id] = {}
        if (head, tail) not in docid2vecs[doc_id]:
            docid2vecs[doc_id][(head, tail)] = np.zeros(dim).astype(np.float16)
        docid2vecs[doc_id][(head, tail)] += rel
    
    return docid2vecs
