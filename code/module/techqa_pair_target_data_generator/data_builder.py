import os

import csv
import json
import jsonlines
import random
import re
from tqdm import tqdm

from ..utils.entity_normalizer import \
    normalize_techqa_string, \
    TechQATokenizer

class DataBuilder:
    def __init__(self, args):
        self.args = args
        self.funcs = []
        if self.args.preprocess_qa_pairs:
            self.funcs.append(self.preprocess_qa_pairs)
        if self.args.get_subset:
            self.funcs.append(self.get_subset_of_train_set)
    
    def run(self):
        for func in self.funcs:
            func()
        return 0
    
    def preprocess_qa_pairs(self):
        assert self.args.techqa_qa_file != None \
            and self.args.entity_file != None \
            and self.args.q_doc_file != None
        
        with jsonlines.open(self.args.entity_file, "r") as reader:
            e2id = {
                r["entity"]: r["entity_id"] for r in reader
            }
            total_entities = set([e for e in e2id])
        
        with open(self.args.techqa_qa_file, "r") as fr:
            qas = json.load(fr)

        err = 0
        techqa_doc_tokenier = TechQATokenizer(e2id)
        os.makedirs(os.path.dirname(self.args.q_doc_file), exist_ok=True) 
        with jsonlines.open(self.args.q_doc_file, "w") as writer:
            for qa in tqdm(qas, desc="Reading {}".format(self.args.techqa_qa_file.split('/')[-1])):
                qid = qa["QUESTION_ID"]
                title = qa["QUESTION_TITLE"] 
                text = qa["QUESTION_TEXT"]
                q = "{} {}".format(title, text)
                
                result = techqa_doc_tokenier \
                            .techqa_entity_matching_and_tokenizing(q)
                tokens = result["tokens"]
                entities = result["entities"]
                target_entity_pairs = [ \
                    (i, j) for i, e1 in enumerate(entities) \
                        for j, e2 in enumerate(entities) \
                            if e1[1] != e2[1] \
                ]
                if len(target_entity_pairs) == 0:
                    err += 1
                example = {
                    "qid": qid,
                    "question_tokens": tokens,
                    "mentions": entities,
                    "target_entity_pairs": target_entity_pairs,
                    "answer": qa["ANSWER"],
                    "doc_id": qa["DOCUMENT"],
                    "answerable": qa["ANSWERABLE"],
                    "candidate_doc_ids": qa["DOC_IDS"]
                }
                writer.write(example)
        print("N Err: {}".format(err))
        return 0
    
    def get_subset_of_train_set(self):
        assert self.args.q_doc_file != None \
            and self.args.proportion != None \
            and self.args.q_doc_subset_file != None

        with jsonlines.open(self.args.q_doc_file, "r") as reader:
            data = [r for r in reader]

        n_samples = int(self.args.proportion * len(data))
        new_data = random.sample(data, n_samples)
        os.makedirs(os.path.dirname(self.args.q_doc_subset_file), exist_ok=True) 
        with jsonlines.open(self.args.q_doc_subset_file, "w") as writer:
            for d in new_data:
                writer.write(d)
        return 0
