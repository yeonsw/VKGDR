import os

import collections
from collections import namedtuple
import csv
import cupy
from itertools import cycle
from joblib import Parallel, delayed
import json
import jsonlines
import math
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import nltk
import numpy as np
import random
import re
import spacy
import string
import torch
from tqdm import tqdm

from ..utils.entity_normalizer import \
    normalize_techqa_string, \
    TechQATokenizer

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def process_doc(doc):
    tokens = []
    mentions = []
    for token in doc:
        if token.ent_iob_ == "B":
            mentions.append(len(tokens))
            tokens.append(token.text)
            continue
        if token.ent_iob_ == "I":
            tokens[-1] = "{} {}".format(tokens[-1], token.text)
            continue
        if token.ent_iob_ == "O":
            tokens.append(token.text)
    return (tokens, mentions)

def process_chunk(texts, rank):
    spacy.prefer_gpu(rank)
    nlp = spacy.load("en_core_web_sm")
    preproc_pipe = []
    for doc in tqdm(nlp.pipe(texts, batch_size=512), total=len(texts), desc="Running Spacy"):
        tokens, mentions = process_doc(doc)
        preproc_pipe.append((tokens, mentions))
    rank+=1
    return preproc_pipe

def run_spacy_parallel(texts):
    n_gpus = cupy.cuda.runtime.getDeviceCount()
    chunksize = math.ceil(len(texts) / n_gpus)
    executor = Parallel(n_jobs=n_gpus, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = []
    gpus = list(range(0, n_gpus))
    rank = 0
    for chunk in chunker(texts, len(texts), chunksize=chunksize):
        tasks.append(do(chunk, rank))
        rank = (rank+1)%len(gpus)
    result = executor(tasks)
    return flatten(result)

class TechQAPreprocessor:
    def __init__(self, args):
        self.args = args
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

        self.funcs = []
        if self.args.building_predefined_entities_list:
            self.funcs.append(self.building_predefined_entities_multi_gpus)
        if self.args.preprocess_techqa:
            self.funcs.append(self.preprocess_techqa)
        if self.args.split_train_valid:
            self.funcs.append(self.split_train_valid)

    def run(self):
        for func in self.funcs:
            func()
        return 0
    
    def read_techqa_corpus(self, inpf):
        corpus = None
        if inpf.endswith(".jsonl"):
            corpus = self.read_techqa_corpus_jsonl(inpf)
        elif inpf.endswith(".json"):
            corpus = self.read_techqa_corpus_json(inpf)
        else:
            raise Exception("The format of the input file should be jsonl or json")
        return corpus
   
    def preprocess_single_doc(self, passage):
        snippets = []
        
        passage_tokens = passage["text"].split()
        passage_size = 64
        n_passages = math.ceil(1.0 * len(passage_tokens)/passage_size)
        chunked_passages = []
        for pind in range(n_passages):
            s = passage_size * pind
            e = min(passage_size * (pind + 1), len(passage_tokens))
            chunked_passage = " ".join(passage_tokens[s:e])
            snippets.append({ \
                "id": passage["id"],
                "title": passage["title"],
                "text": chunked_passage,
                "metadata": passage["metadata"],
                "chunk_ind": pind,
            })
        return snippets

    def read_techqa_corpus_jsonl(self, inpf):
        # Reading the TechQA corpus
        corpus = []
        with jsonlines.open(inpf, "r") as reader:
            for passage in tqdm(reader, desc="Reading TechQA passages"):
                corpus += self.preprocess_single_doc(passage)
        return corpus
    
    def read_techqa_corpus_json(self, inpf):
        # Reading the TechQA corpus
        corpus = []
        with open(inpf, "r") as fr:
            data = json.load(fr)
            for passage_id in tqdm(data, desc="Reading TechQA passages"):
                passage = data[passage_id]
                corpus += self.preprocess_single_doc(passage)
                
        return corpus
    
    def building_predefined_entities_multi_gpus(self):
        assert self.args.techqa_corpus_file != None \
            and self.args.entity_file != None
        
        corpus = self.read_techqa_corpus(self.args.techqa_corpus_file)
        # Building pre-defined entities
        passages = run_spacy_parallel([d["text"] for d in corpus])
        
        entity2id = {}
        entity_id = 0
        os.makedirs(os.path.dirname(self.args.entity_file), exist_ok=True) 
        with jsonlines.open(self.args.entity_file, "w") as writer:
            for tokens, mentions in tqdm(passages, \
                desc="Building pre-defined entities" \
            ):
                for mind in mentions:
                    entity = normalize_techqa_string(tokens[mind])
                    if entity == "":
                        continue
                    if len(entity) < 2:
                        continue
                    if entity in self.stopwords:
                        continue
                    if entity not in entity2id:
                        entity2id[entity] = str(entity_id)
                        entity_id += 1
                        
                        writer.write({
                            "entity_id": entity2id[entity],
                            "entity": entity,
                        })
               
        return 0
    
    def building_predefined_entities(self):
        assert self.args.techqa_corpus_file != None \
            and self.args.entity_file != None
        
        spacy.prefer_gpu()
        spacy_pipeline = spacy.load("en_core_web_sm")
        
        corpus = self.read_techqa_corpus(self.args.techqa_corpus_file)
        # Building pre-defined entities
        passages = spacy_pipeline.pipe( \
            [d["text"] for d in corpus], \
            batch_size=256 \
        )
        
        entity2id = {}
        entity_id = 0
        os.makedirs(os.path.dirname(self.args.entity_file), exist_ok=True) 
        with jsonlines.open(self.args.entity_file, "w") as writer:
            for c, p in tqdm( \
                zip(corpus, passages), \
                total=len(corpus), \
                desc="Building pre-defined entities" \
            ):
                tokens = []
                mentions = []
                for token in p:
                    if token.ent_iob_ == "B":
                        mentions.append(len(tokens))
                        tokens.append(token.text)
                        continue
                    if token.ent_iob_ == "I":
                        tokens[-1] = "{} {}".format(tokens[-1], token.text)
                        continue
                    if token.ent_iob_ == "O":
                        tokens.append(token.text)
                
                local_entities = []
                for mind in mentions:
                    entity = normalize_techqa_string(tokens[mind])
                    if entity == "":
                        continue
                    if len(entity) < 2:
                        continue
                    if entity in self.stopwords:
                        continue
                    if entity not in entity2id:
                        entity2id[entity] = str(entity_id)
                        entity_id += 1
                        
                        writer.write({
                            "entity_id": entity2id[entity],
                            "entity": entity,
                        })
                    local_entities.append(entity)
               
                local_entities = list(set(local_entities))
        return 0

    def get_corpus_stats(self, preprocessed_docs):
        assert self.args.target_entity_pair_file != None \
            and self.args.eid2n_file != None 
        #build eid2n
        eid2n = collections.defaultdict(int)
        for result in tqdm(preprocessed_docs, \
            desc="Building eid2n" \
        ):
            tokens = result["tokens"]
            mentions = result["entities"]
            local_entities = set([eid for _, eid in mentions])
            for eid in local_entities:
                eid2n[eid] += 1
        
        os.makedirs(os.path.dirname(self.args.eid2n_file), exist_ok=True) 
        with jsonlines.open(self.args.eid2n_file, "w") as writer:
            for eid in tqdm(eid2n, desc="Writing eid2n"):
                writer.write({"entity_id": eid, "n": eid2n[eid]})
        
        # Build pair2n
        pair2n = collections.defaultdict(int)
        for result in tqdm(preprocessed_docs, \
            desc="Computing pair2n", \
        ):
            filtered_mentions = [ \
                {"idx": i, "mention": m} for i, m in enumerate(result["entities"]) \
                    if eid2n[m[1]] >= self.args.n_entity_thresh \
            ]
            uniq_mentions = list(set([
                elem["mention"][1] for elem in filtered_mentions \
            ]))
            for elem1 in uniq_mentions:
                for elem2 in uniq_mentions:
                    if elem1 != elem2:
                        pair2n[(elem1, elem2)] += 1
        
        os.makedirs(os.path.dirname(self.args.target_entity_pair_file), exist_ok=True) 
        with jsonlines.open( \
            self.args.target_entity_pair_file, "w" \
        ) as writer:
            n = len(preprocessed_docs)
            for pair in tqdm(pair2n, desc="Saving pmi scores"):
                e1, e2 = pair
                pmi = np.log(pair2n[pair] / n) \
                        - np.log(eid2n[e1] / n) \
                        - np.log(eid2n[e2] / n)
                writer.write({
                    "head": e1,
                    "tail": e2,
                    "pmi": "{:.04f}".format(pmi),
                    "n": "{}".format(pair2n[pair])
                })
        return eid2n

    def preprocess_techqa(self):
        assert self.args.techqa_corpus_file != None \
            and self.args.entity_file != None \
            and self.args.techqa_corpus_preprocessed_file != None \
            and self.args.get_stats != None \
            and self.args.eid2n_file != None
        
        corpus = self.read_techqa_corpus(self.args.techqa_corpus_file)
        with jsonlines.open(self.args.entity_file, "r") as reader:
            entity2id = {}
            entityid2entity = {}
            for r in tqdm(reader, desc="Reading the entity file"):
                entity2id[r["entity"]] = r["entity_id"]
                entityid2entity[r["entity_id"]] = r["entity"]

        # Preprocessing docs
        techqa_doc_tokenier = TechQATokenizer(entity2id)
        
        #Tokenizing
        preprocessed_docs = []
        for doc in tqdm(corpus, \
            desc="Tokenizing TechQA passages" \
        ):
            doc_text = None
            if self.args.use_title:
                doc_text = "{} {}".format(doc["title"], doc["text"])
            else:
                doc_text = "{}".format(doc["text"])

            result = techqa_doc_tokenier.techqa_entity_matching_and_tokenizing(doc_text)
            preprocessed_docs.append(result)
        
        eid2n = None
        if self.args.get_stats:
            eid2n = self.get_corpus_stats(preprocessed_docs)
        else:
            eid2n = {}
            with jsonlines.open(self.args.eid2n_file, "r") as reader:
                for r in tqdm(reader, desc="Writing eid2n"):
                    eid2n[r["entity_id"]] = r["n"]
        entity2id = {}
        for eid in tqdm(eid2n, desc="Building entity2id"):
            entity2id[entityid2entity[eid]] = eid

        os.makedirs(os.path.dirname(self.args.techqa_corpus_preprocessed_file), exist_ok=True) 
        with jsonlines.open(self.args.techqa_corpus_preprocessed_file, 'w') as fw:
            for doc, result in tqdm(zip(corpus, preprocessed_docs), \
                desc="Preprocessing TechQA passages", \
                total=len(corpus), \
            ):
                target_entity_pairs = []
                filtered_mentions = [ \
                    {"idx": i, "mention": m} for i, m in enumerate(result["entities"]) \
                        if eid2n[m[1]] >= self.args.n_entity_thresh \
                ]
                for elem1 in filtered_mentions:
                    h, _ = elem1["mention"]
                    i = elem1["idx"]
                    for elem2 in filtered_mentions:
                        t, _ = elem2["mention"]
                        j = elem2["idx"]
                        if h != t:
                            target_entity_pairs += [ \
                                (i, j) \
                            ]
            
                example = { 
                    "id": doc["id"], 
                    'title': doc["title"], 
                    'text': doc["text"],
                    'metadata': doc["metadata"],
                    "sentence": result["tokens"],
                    "entities": result["entities"],
                    "target_entity_pairs": target_entity_pairs,
                    "chunk_ind": doc["chunk_ind"]
                }
                fw.write(example)
        return 0

    def split_train_valid(self):
        assert self.args.techqa_corpus_preprocessed_file != None \
            and self.args.train_file != None \
            and self.args.valid_file != None \
            and self.args.train_prop != None

        os.makedirs(os.path.dirname(self.args.train_file), exist_ok=True) 
        os.makedirs(os.path.dirname(self.args.valid_file), exist_ok=True) 
        with jsonlines.open(self.args.techqa_corpus_preprocessed_file, "r") as reader:
            with jsonlines.open(self.args.train_file, "w") as train_writer:
                with jsonlines.open(self.args.valid_file, "w") as valid_writer:
                    for r in tqdm(reader, desc="Spliting train valid"):
                        if np.random.rand() < self.args.train_prop:
                            train_writer.write(r)
                        else:
                            valid_writer.write(r)
        return 0 
