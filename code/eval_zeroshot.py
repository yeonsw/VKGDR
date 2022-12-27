import os

import argparse
import jsonlines
import numpy as np
import random
from scipy.stats import rankdata
from scipy import stats
from tabulate import tabulate
from tqdm import tqdm

from module.utils.vkg_utils import read_vkg
from module.utils.file_utils import read_bm25_score_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_q_file", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--vkg_path", type=str, required=True)
    
    parser.add_argument("--bm25_file", type=str, required=True)
    parser.add_argument("--rel_lmb", type=float, required=True)
    parser.add_argument("--bm25_lmb", type=float, required=True)

    args = parser.parse_args()
    return args

class DocRetriever:
    def __init__(self, config):
        self.config = config
        
        self.bm25_results = read_bm25_score_file(self.config.bm25_file)
        self.docid2vkg = read_vkg( \
            self.config.vkg_path \
        )
        self.qid2cands = {}
        with jsonlines.open(self.config.original_q_file, "r") as reader:
            for r in tqdm(reader, desc="Reading original q file"):
                if r["candidate_doc_ids"] != None:
                    self.qid2cands[r["qid"]] = r["candidate_doc_ids"]
                else:
                    bm25_doc_ids_scores = list(self.bm25_results[r["qid"]].items())
                    bm25_doc_ids_scores.sort(key=lambda x: x[1], reverse=True)
                    self.qid2cands[r["qid"]] = [doc_id for doc_id, _ in bm25_doc_ids_scores[:50]]
                    if r["doc_id"] not in self.qid2cands[r["qid"]]:
                        self.qid2cands[r["qid"]][-1] = r["doc_id"]

    def get_doc_vecs(self, doc_id):
        doc_vec = {}
        if doc_id not in self.docid2vkg:
            return doc_vec
        
        for pair, vec in zip(self.docid2vkg[doc_id]["head_tails"], self.docid2vkg[doc_id]["vecs"]):
            e_pair = tuple(pair)
            if e_pair not in doc_vec:
                doc_vec[e_pair] = {
                    "vecs": [],
                }
            doc_vec[e_pair]["vecs"].append(vec)
        for e_pair in doc_vec:
            doc_vec[e_pair]["vec"] = np.sum(doc_vec[e_pair]["vecs"], axis=0)
        return doc_vec
    
    def compute_rank(self, qid, q, gid, docid2vecs):
        def _compute_similarity(q, doc):
            rel_sim = 0.0
            for pair in q:
                q_vecs = np.array(q[pair]["vec"])
                if pair not in doc:
                    continue
                d_vecs = np.array(doc[pair]["vec"])
                score = np.dot(q_vecs, d_vecs)
                rel_sim += score
            
            return rel_sim
        
        similarities = []
        g_ind = 0
        for i, doc_id in enumerate(docid2vecs):
            rel_sim = _compute_similarity(q, docid2vecs[doc_id])
            bm25_sim = self.bm25_results[qid][doc_id] \
                if doc_id in self.bm25_results[qid] else 0.0
            similarities.append((doc_id, bm25_sim, rel_sim))
            if doc_id == gid:
                g_ind = i
        
        gs = similarities.pop(g_ind)
        similarities = similarities + [gs]
        
        doc_ids = [s[0] for s in similarities]
        
        bm25_sim = [s[1] for s in similarities]
        bm25_rank = rankdata(-np.array(bm25_sim), method='min')
        
        rel_sim = [s[2] for s in similarities]
        rel_rank = rankdata(-np.array(rel_sim), method='min') 
        
        weighted_sum = - 1.0 * ( \
            self.config.rel_lmb * rel_rank + self.config.bm25_lmb * bm25_rank \
        )
        
        similarities = [(doc_id, b, r, w) for doc_id, b, r, w in zip(doc_ids, bm25_sim, rel_sim, weighted_sum)]
        
        bm25_similarities = [ \
            (doc_id, s) for doc_id, s, _, __ in \
                sorted(similarities, key=lambda x: x[1], reverse=True) \
        ]
        rel_similarities = [ \
            (doc_id, s) for doc_id, _, s, __ in \
                sorted(similarities, key=lambda x: x[2], reverse=True) \
        ]
        weighted_similarities = [ \
            (doc_id, s) for doc_id, _, __, s in \
                sorted(similarities, key=lambda x: x[3], reverse=True) \
        ]
        
        merge_similarities = []
        seen = set([])
        for i in range(len(weighted_similarities)):
            if weighted_similarities[i][0] not in seen:
                merge_similarities.append(weighted_similarities[i])
                seen.add(weighted_similarities[i][0])
            if bm25_similarities[i][0] not in seen:
                merge_similarities.append(bm25_similarities[i]) 
                seen.add(bm25_similarities[i][0])

        def _get_rank(sims, gid):
            rank = len(sims)
            for i, s in enumerate(sims):
                if s[0] == gid:
                    rank = i + 1
                    break
            return rank
        
        weighted_rank = _get_rank(weighted_similarities, gid)
        rel_rank = _get_rank(rel_similarities, gid)
        bm25_rank = _get_rank(bm25_similarities, gid)
        merge_rank = _get_rank(merge_similarities, gid)
        
        return {
            "bm25_rank": bm25_rank, 
            "rel_rank": rel_rank, 
            "weighted_rank": weighted_rank,
            "merge_rank": merge_rank,
        }

    def retrieve(self, rel_lmb, bm25_lmb):
        weighted_1 = []
        rel_1 = []
        bm25_1 = []
        merge_1 = []
        
        weighted_5 = []
        rel_5 = []
        bm25_5 = []
        merge_5 = []
        
        rel_mrr = []
        bm25_mrr = []
        weighted_mrr = []
        merge_mrr = []
        
        with jsonlines.open(self.config.query_file, "r") as reader:
            queries = [r for r in reader if r["gt"] != "-"]
        
        for d in tqdm(queries, desc="Processing"):
            gid = d["gt"]
            qid = d["qid"]
            
            query = {}
            for pair, vec in zip(d["q_entity_pair"], d["qvec"]):
                e_pair = tuple(pair)
                if e_pair not in query:
                    query[e_pair] = {
                        "vecs": [],
                    }
                query[e_pair]["vecs"].append(vec)
            
            for e_pair in query:
                query[e_pair]["vec"] = np.sum(query[e_pair]["vecs"], axis=0)

            docid2vecs = {}
            for doc_id in self.qid2cands[d["qid"]]:
                docid2vecs[doc_id] = self.get_doc_vecs(doc_id)
            
            preds = self.compute_rank(qid, query, gid, docid2vecs)
            
            weighted_1.append(1.0 if preds["weighted_rank"] == 1 else 0.0)
            rel_1.append(1.0 if preds["rel_rank"] == 1 else 0.0)
            merge_1.append(1.0 if preds["merge_rank"] == 1 else 0.0)
            bm25_1.append(1.0 if preds["bm25_rank"] == 1 else 0.0)
                
            weighted_5.append(1.0 if preds["weighted_rank"] <= 5 else 0.0)
            rel_5.append(1.0 if preds["rel_rank"] <= 5 else 0.0)
            merge_5.append(1.0 if preds["merge_rank"] <= 5 else 0.0)
            bm25_5.append(1.0 if preds["bm25_rank"] <= 5 else 0.0)
            
            weighted_mrr.append(1.0 / preds["weighted_rank"])
            rel_mrr.append(1.0 / preds["rel_rank"])
            merge_mrr.append(1.0 / preds["merge_rank"])
            bm25_mrr.append(1.0 / preds["bm25_rank"])
            
            print("\n")
            print("*" * 10)
            table = [ \
                ["Model", "R@1", "R@5", "MRR"], \
                ["VKGDR", 100 * np.mean(rel_1), 100 * np.mean(rel_5), 100 * np.mean(rel_mrr)], \
                ["BM25", 100 * np.mean(bm25_1), 100 * np.mean(bm25_5), 100 * np.mean(bm25_mrr)], \
                ["Weight ({})".format(bm25_lmb), 100 * np.mean(weighted_1), 100 * np.mean(weighted_5), 100 * np.mean(weighted_mrr)], \
                ["MERGE", 100 * np.mean(merge_1), 100 * np.mean(merge_5), 100 * np.mean(merge_mrr)], \
            ]
            print(tabulate(table, headers='firstrow', tablefmt='grid'))
        return 0

def main(args):
    dr = DocRetriever(args)
    dr.retrieve(args.rel_lmb, args.bm25_lmb)

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
