#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dataclasses import dataclass, field
import logging
import random
import json
import jsonlines
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from tqdm import tqdm
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class QADataTrainingArguments:
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    corpus_file: Optional[str] = field(default=None)
    target_entity_pair_file: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=128)
    n_hard_negs: Optional[int] = field(default=8)
    sampling_ratio: Optional[float] = field(default=None)
    pmi_thresh: Optional[float] = field(default=5.0)
    n_pair_thresh: Optional[int] = field(default=10)

    def __post_init__(self):
        if self.train_file is None or self.eval_file is None:
            raise ValueError("Need either a training/evalation file.")

class QADataset(Dataset):
    def __init__(self, question_file, corpus_file, target_entity_pair_file, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length \
            = self.config.max_length
        self.q_data, self.qid2docids, self.doc_data, self.docid2inds = self.get_data( \
            question_file, \
            corpus_file, \
            target_entity_pair_file, \
            self.config.pmi_thresh, \
            self.config.n_pair_thresh \
        )
        assert len(self.q_data) > 0
    
    def get_masked_sentences_from_sentence(self, inp, all_target_entity_pairs, sentence_type):
        tokens = inp["sentence"] if sentence_type == "document" else inp["question_tokens"]
        entities = inp["entities"] if sentence_type == "document" else inp["mentions"]
        target_pairs = inp["target_entity_pairs"]
        doc_id = inp["id"] if sentence_type == "document" else inp["qid"]
        doc_chunk_ind = inp["chunk_ind"] if sentence_type == "document" else 0
        
        entity_pairs = []
        if target_pairs != "all":
            for e1_idx, e2_idx in target_pairs:
                e1 = entities[e1_idx]
                e2 = entities[e2_idx]
                if sentence_type == "document":
                    if (e1[1], e2[1]) in all_target_entity_pairs:
                        entity_pairs.append((e1, e2))
                    else:
                        continue
                elif sentence_type == "question":
                    entity_pairs.append((e1, e2))
                else:
                    raise Exception("Invalid sentence type")
                    
        else:
            for e1 in entities:
                for e2 in entities:
                    if e1[1] != e2[1] \
                        and (e1[1], e2[1]) in all_target_entity_pairs:
                        entity_pairs.append((e1, e2))
        
        data = []
        for head_e, tail_e in entity_pairs:
            head = head_e[0]
            tail = tail_e[0]
            pairs = [ \
                (head, "[unused1]"), (tail, "[unused2]") \
            ]
            pairs.sort( \
                key=lambda x: x[0], \
                reverse=True \
            )
            
            new_sentence = tokens[:]
            for target_ind, sp_token in pairs:
                new_sentence.insert( \
                    target_ind + 1, sp_token \
                )
                new_sentence[target_ind] = "[unused3]"
            new_sentence = " ".join( \
                new_sentence \
            )
            data.append({
                "sentence": new_sentence,
                "chunk_ind": doc_chunk_ind,
                "id": doc_id,
                "entity_pair": (
                    head_e[1], tail_e[1], \
                )
            })
            if sentence_type == "question":
                data[-1]["answer_id"] = inp["doc_id"]
        return data
    
    def read_corpus(self, corpus_file):
        docs = []
        with jsonlines.open(corpus_file, "r") as reader:
            for r in tqdm(reader, desc="Reading {}".format(corpus_file.split('/')[-1])):
                if np.random.rand() < self.config.sampling_ratio:
                    docs.append(r)
        return docs
    
    def read_target_pair_file(self, target_pair_file, pmi_thresh, n_pair_thresh):
        target_pairs = set([])
        with jsonlines.open(target_pair_file, "r") as reader:
            for r in tqdm(reader, desc="Reading target pair file"):
                if "pmi" in r and float(r["pmi"]) < pmi_thresh:
                    continue
                if "n" in r and float(r["n"]) < n_pair_thresh:
                    continue
                target_pairs.add((r["head"], r["tail"]))
        print("N Target pairs: {}".format(len(target_pairs))) 
        return target_pairs

    def read_qfile(self, qfile):
        qs = []
        qid2docid = {}
        with jsonlines.open(qfile, "r") as reader:
            for r in tqdm(reader, desc="Reading {}".format(qfile.split('/')[-1])):
                if r["answerable"] == "Y":
                    qs.append(r)
                    qid2docid[r["qid"]] = r["candidate_doc_ids"]
        return (qs, qid2docid)

    def get_data(self, q_file, corpus_file, target_pair_file, pmi_thresh=5.0, n_pair_thresh=10):
        target_pairs = self.read_target_pair_file(target_pair_file, pmi_thresh, n_pair_thresh)
        docs = self.read_corpus(corpus_file)
        doc_data, docid2inds = self.preprocessing_docs(docs, target_pairs, sentence_type="document")

        qs, qid2docids = self.read_qfile(q_file)
        q_data, _ = self.preprocessing_docs(qs, target_pairs, sentence_type="question")
        
        #Filter questions without answers
        q_data = [q for q in q_data if q["answer_id"] in docid2inds]
        return (q_data, qid2docids, doc_data, docid2inds)
    
    def preprocessing_docs(self, docs, target_pairs, sentence_type):
        r1_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused1]')
        r2_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused2]')
        
        print("Tokenizing")
        err = 0
        data = []
        chunk_size = 100
        progress_bar = tqdm(range(0, len(docs), chunk_size), desc="Generating new data")
        for s in progress_bar:
            progress_bar.set_description( \
                "# new data / err ({:d}/{:d})" \
                    .format(len(data), err) \
            )
            target_docs = docs[s:s+chunk_size]
            
            masked_sentences = []
            for doc in target_docs:
                masked_sentences += self.get_masked_sentences_from_sentence(doc, target_pairs, sentence_type)
            if len(masked_sentences) == 0:
                continue
            
            target_sentences = [d["sentence"] for d in masked_sentences]
            tokenized = self.tokenizer.batch_encode_plus( \
                target_sentences, \
                padding="max_length", \
                max_length=self.max_length, \
                truncation=True, \
                return_tensors='pt' \
            )
            
            input_ids = tokenized["input_ids"]
            token_type_ids = tokenized["token_type_ids"]
            attention_mask = tokenized["attention_mask"]
            
            for ind, iids in enumerate(input_ids):
                r1_idx = torch.nonzero(
                    (iids == r1_token)
                ).flatten().tolist()
                r2_idx = torch.nonzero(
                    (iids == r2_token)
                ).flatten().tolist()
                
                if len(r1_idx) == 0 \
                    or len(r2_idx) == 0:
                    err += 1
                    continue
                data.append(masked_sentences[ind])
        print( \
            "# new data/err: {:d}/{:d}" \
                .format(len(data), err) \
        )
        
        doc_id2inds = collections.defaultdict(list)
        for i, d in enumerate(tqdm(data, desc="Generating helper")):
            doc_id2inds[d["id"]].append(i)
        return (data, doc_id2inds)
    
    def tokenize(self, sentences):
        r1_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused1]')
        r2_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused2]')
        tokenized = self.tokenizer.batch_encode_plus( \
            sentences, \
            padding="max_length", \
            max_length=self.max_length, \
            truncation=True, \
            return_tensors='pt' \
        )
        data = []
        for input_idx in range(len(sentences)):
            input_ids = tokenized["input_ids"][input_idx]
            r1_idx = torch.nonzero(
                (input_ids == r1_token)
            ).flatten().tolist()
            r2_idx = torch.nonzero(
                (input_ids == r2_token)
            ).flatten().tolist()
            data.append({
                "head_idx": r1_idx[0],
                "tail_idx": r2_idx[0]
            })
            features = list(tokenized.keys())
            for key in features:
                data[-1][key] = tokenized[key][input_idx]
        return data

    def __len__(self):
        return len(self.q_data)

    def __getitem__(self, idx):
        doc_id = self.q_data[idx]["id"]
        answer_id = self.q_data[idx]["answer_id"]
        chunk_ind = self.q_data[idx]["chunk_ind"]
        sentence = self.q_data[idx]["sentence"]
        entity_pair = self.q_data[idx]["entity_pair"]

        """
        Get positive sample
        """
        answer_inds = self.docid2inds[answer_id]
        pos_ind = random.choice(answer_inds)
        pos_sample = self.doc_data[pos_ind]

        #Negative sample
        neg_doc_ids = self.qid2docids[doc_id]
        if answer_id in neg_doc_ids:
            neg_doc_ids.remove(answer_id)
        
        neg_doc_ids = [ \
            ndoc for ndoc in neg_doc_ids \
                if len(self.docid2inds[ndoc]) > 0]
        n_negs = min(len(neg_doc_ids), self.config.n_hard_negs)
        negs = random.sample( \
            neg_doc_ids, \
            n_negs \
        )
        neg_docs = [ \
            self.doc_data[random.choice(self.docid2inds[neg])] for neg in negs \
        ]
        neg_sentences = [elem["sentence"] for elem in neg_docs]

        all_sentences = [sentence, pos_sample["sentence"]] + neg_sentences
        all_data = self.tokenize(all_sentences)
        
        q_data = all_data[:1]
        pos_data = all_data[1:2]
        neg_data = all_data[2:]
        
        q_data[0]["entity_pair"] = self.q_data[idx]["entity_pair"]
        q_data[0]["sentence"] = self.q_data[idx]["sentence"]
        q_data[0]["doc_id"] = doc_id
        q_data[0]["answer_id"] = answer_id
        
        pos_data[0]["entity_pair"] = pos_sample["entity_pair"]
        pos_data[0]["sentence"] = pos_sample["sentence"]
        pos_data[0]["doc_id"] = pos_sample["id"]

        for neg, neg_doc in zip(neg_data, neg_docs):
            neg["entity_pair"] = neg_doc["entity_pair"]
            neg["sentence"] = neg_doc["sentence"]
            neg["doc_id"] = neg_doc["id"]
        
        return {
            "input": q_data[0],
            "pos_sample": pos_data,
            "neg_sample": neg_data,
        }

def qa_data_collator(samples):
    if len(samples) == 0:
        return {}
    bsize = len(samples) 
    inputs = []
    positives = []
    negatives = []
    for sample in samples:
        inputs.append(sample["input"])
        positives += sample["pos_sample"]
        negatives += sample["neg_sample"]
    
    assert len(inputs) == len(positives)
    #sample, positives
    all_samples = inputs + positives + negatives
    pos_indices = list( \
        range( \
            len(inputs), \
            len(inputs) + len(positives) \
        ) \
    )
    
    batch_mask_indices = []
    for pos_ind, inp in zip(pos_indices, inputs):
        answer_id = inp["answer_id"]
        mask_indices = [0] * len(all_samples)
        for inst_idx, inst in enumerate(all_samples):
            inst_doc_id = inst["doc_id"]
            if inst_idx == pos_ind:
                continue
            if inst_doc_id == answer_id:
                mask_indices[inst_idx] = 1
        batch_mask_indices.append(mask_indices)
    
    pos_neg = positives + negatives
    pos_neg_input_ids = []
    pos_neg_token_type_ids = []
    pos_neg_attention_mask = []
    pos_neg_head_indices = []
    pos_neg_tail_indices = []
    for inst in pos_neg:
        pos_neg_input_ids.append( \
            inst["input_ids"] \
        )
        pos_neg_token_type_ids.append( \
            inst["token_type_ids"] if "token_type_ids" in inst else None \
        )
        pos_neg_attention_mask.append( \
            inst["attention_mask"] \
        )
        pos_neg_head_indices.append( \
            inst["head_idx"]
        )
        pos_neg_tail_indices.append( \
            inst["tail_idx"]
        )
    pos_neg_input_ids = torch.stack( \
        pos_neg_input_ids, dim=0 \
    )
    if pos_neg_token_type_ids[0] != None:
        pos_neg_token_type_ids = torch.stack( \
            pos_neg_token_type_ids, dim=0 \
        )
    else:
        pos_neg_token_type_ids = None
        
    pos_neg_attention_mask = torch.stack( \
        pos_neg_attention_mask, dim=0 \
    )
    pos_neg_head_indices = torch.tensor( \
        pos_neg_head_indices, dtype=torch.long \
    )
    pos_neg_tail_indices = torch.tensor( \
        pos_neg_tail_indices, dtype=torch.long \
    )
    
    input_input_ids = []
    input_token_type_ids = []
    input_attention_mask = []
    input_head_indices = []
    input_tail_indices = []
    for inp in inputs:
        input_input_ids.append( \
            inp["input_ids"] \
        )
        input_token_type_ids.append( \
            inp["token_type_ids"] if "token_type_ids" in inp else None\
        )
        input_attention_mask.append( \
            inp["attention_mask"] \
        )
        input_head_indices.append( \
            inp["head_idx"]
        )
        input_tail_indices.append( \
            inp["tail_idx"]
        )
    input_input_ids = torch.stack(input_input_ids, dim=0)
    if input_token_type_ids[0] != None:
        input_token_type_ids = torch.stack( \
            input_token_type_ids, dim=0 \
        )
    else:
        input_token_type_ids = None
    input_attention_mask = torch.stack( \
        input_attention_mask, dim=0 \
    )
    input_head_indices = torch.tensor( \
        input_head_indices, dtype=torch.long \
    )
    input_tail_indices = torch.tensor( \
        input_tail_indices, dtype=torch.long \
    )
    batch_mask_indices = torch.tensor( \
        batch_mask_indices, dtype=torch.long \
    )
    pos_indices = torch.tensor( \
        pos_indices, dtype=torch.long \
    )
    
    input_ids = torch.cat((input_input_ids, pos_neg_input_ids))
    
    token_type_ids = None
    if input_token_type_ids != None and pos_neg_token_type_ids != None:
        token_type_ids = torch.cat( \
            (input_token_type_ids, pos_neg_token_type_ids) \
        )
    
    attention_mask = torch.cat((input_attention_mask, pos_neg_attention_mask))
    head_indices = torch.cat((input_head_indices, pos_neg_head_indices))
    tail_indices = torch.cat((input_tail_indices, pos_neg_tail_indices))
    new_batch = {
        "n_samples": bsize,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "head_indices": head_indices,
        "tail_indices": tail_indices,
        "pos_indices": pos_indices,
        "batch_mask_indices": batch_mask_indices
    }
    if token_type_ids != None:
        new_batch["token_type_ids"] = token_type_ids
    return new_batch 
