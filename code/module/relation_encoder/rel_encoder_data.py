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
class RelPredDataArguments:
    corpus_file: Optional[str] = field(default=None)
    target_pair_file: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=128)

    def __post_init__(self):
        if self.corpus_file is None:
            raise ValueError("Need a corpus file.")

class VKGDRRelPredDataset(Dataset):
    def __init__(self, input_file, target_pair_file, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length \
            = self.config.max_length
        self.data = self.get_data(input_file, target_pair_file)
    
    def get_masked_sentences_from_sentence(self, inp, all_target_entity_pairs):
        tokens = inp["sentence"]
        entities = inp["entities"]
        target_pairs = inp["target_entity_pairs"]

        entity_pairs = []
        if target_pairs != "all":
            for e1_idx, e2_idx in target_pairs:
                e1 = entities[e1_idx]
                e2 = entities[e2_idx]
                if (e1[1], e2[1]) in all_target_entity_pairs:
                    entity_pairs.append((e1, e2))
        else:
            for e1 in entities:
                for e2 in entities:
                    if e1[1] != e2[1] \
                        and ((e1[1], e2[1]) in all_target_entity_pairs):
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
            doc_id = inp["id"] if "id" in inp else None
            data.append({
                "doc_id": doc_id, 
                "sentence": new_sentence,
                "entity_pair": (
                    head_e[1], tail_e[1], \
                )
            })
        return data
    
    def get_data(self, fname, target_pair_file):
        with jsonlines.open(fname, "r") as reader:
            sentences = []
            for r in tqdm(reader, desc="Reading {}".format(fname.split('/')[-1])):
                sentences.append(r)
        
        with jsonlines.open(target_pair_file, "r") as reader:
            target_pairs = set([])
            for r in tqdm(reader, desc="Reading target pair file"):
                if "pmi" in r and float(r["pmi"]) < 0.0:
                    continue
                target_pairs.add((r["head"], r["tail"]))
        print("N uniq pairs: {}".format(len(target_pairs)))
        
        data = []
        for sent in tqdm(sentences, desc="Preprocessing sentences"):
            data += self.get_masked_sentences_from_sentence(sent, target_pairs)
        return  data
    
    def tokenize(self, sentence):
        r1_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused1]')
        r2_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused2]')
        tokenized = self.tokenizer( \
            sentence, \
            padding="max_length", \
            max_length=self.max_length, \
            truncation=True, \
            return_tensors='pt' \
        )
        input_ids = tokenized["input_ids"][0]
        
        r1_idx = torch.nonzero(
            (input_ids == r1_token)
        ).flatten().tolist()
        r2_idx = torch.nonzero(
            (input_ids == r2_token)
        ).flatten().tolist()

        if len(r1_idx) == 0 \
            or len(r2_idx) == 0:
            return None

        new_data = {
            "head_idx": r1_idx[0],
            "tail_idx": r2_idx[0]
        }
        features = list(tokenized.keys())
        for key in features:
            new_data[key] = tokenized[key][0]
        return new_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]["sentence"]
        instance = self.tokenize(sentence)
        if instance == None:
            return None
        
        instance["entity_pair"] = self.data[idx]["entity_pair"]
        instance["doc_id"] = self.data[idx]["doc_id"]
        instance["sentence"] = self.data[idx]["sentence"]
        
        return instance

def rel_pred_data_collator(batch_samples):
    samples = [s for s in batch_samples if s != None]
    if len(samples) == 0:
        return {}
    bsize = len(samples)
    entity_pairs = [sample["entity_pair"] for sample in samples]
    doc_ids = [sample["doc_id"] for sample in samples]
    
    input_ids = [inp["input_ids"] for inp in samples]
    input_ids = torch.stack(input_ids, dim=0)
    
    token_type_ids = [inp["token_type_ids"] if "token_type_ids" in inp else None for inp in samples]
    if token_type_ids[0] != None:
        token_type_ids = torch.stack( \
            token_type_ids, dim=0 \
        )
    else:
        token_type_ids = None
    
    attention_mask = [inp["attention_mask"] for inp in samples]
    attention_mask = torch.stack( \
        attention_mask, dim=0 \
    )
    
    head_indices = []
    tail_indices = []
    for inp in samples:
        head_indices.append( \
            inp["head_idx"]
        )
        tail_indices.append( \
            inp["tail_idx"]
        )
    
    head_indices = torch.tensor( \
        head_indices, dtype=torch.long \
    )
    tail_indices = torch.tensor( \
        tail_indices, dtype=torch.long \
    )

    new_batch = {
        "n_samples": bsize,
        "doc_ids": doc_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "head_indices": head_indices,
        "tail_indices": tail_indices,
        "entity_pairs": entity_pairs
    }
    if token_type_ids != None:
        new_batch["token_type_ids"] = token_type_ids
    return new_batch
