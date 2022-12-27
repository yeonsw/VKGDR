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
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    target_entity_pair_file: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=128)
    n_hard_negs: Optional[int] = field(default=8)
    sampling_ratio: Optional[float] = field(default=None)

    def __post_init__(self):
        if self.train_file is None or self.eval_file is None:
            raise ValueError("Need either a training/evalation file.")

class VKGDRDataset(Dataset):
    def __init__(self, input_file, target_entity_pair_file, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length \
            = self.config.max_length
        self.data, self.ht2data_idx, self.head2idx, self.tail2idx = \
            self.get_data(input_file, target_entity_pair_file)
    
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
                "entity_pair": (
                    head_e[1], tail_e[1], \
                )
            })
        return data
    
    def get_data(self, fname, target_pair_file):
        sentences = []
        with jsonlines.open(fname, "r") as reader:
            for r in tqdm(reader, desc="Reading {}".format(fname.split('/')[-1])):
                if np.random.rand() < self.config.sampling_ratio:
                    sentences.append(r)
        
        with jsonlines.open(target_pair_file, "r") as reader:
            target_pairs = set([])
            for r in tqdm(reader, desc="Reading target pair file"):
                if "pmi" in r and float(r["pmi"]) < 5.0:
                    continue
                target_pairs.add((r["head"], r["tail"]))
        print("N Target pairs: {}".format(len(target_pairs))) 
       
        data = [] 
        for sent in tqdm(sentences, desc="Preprocessing sentences"):
            data += self.get_masked_sentences_from_sentence(sent, target_pairs)
        
        r1_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused1]')
        r2_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused2]')

        print("Tokenizing")
        sentences_tokenized = []
        chunk_size = 100000
        
        err = 0
        filtered_data = []
        progress_bar = tqdm(range(0, len(data), chunk_size), desc="Generating new data")
        for s in progress_bar:
            progress_bar.set_description( \
                "# new data / err ({:d}/{:d})" \
                    .format(len(filtered_data), err) \
            )
            
            target_sentences = [d["sentence"] for d in data[s:s+chunk_size]]
            tokenized = self.tokenizer.batch_encode_plus( \
                target_sentences, \
                padding="max_length", \
                max_length=self.max_length, \
                truncation=True, \
                return_tensors='pt' \
            )
            d_inds = list(range(s, s + len(target_sentences)))
            input_ids = tokenized["input_ids"]
            for d_ind, iids in zip(d_inds, input_ids):
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
                filtered_data.append(data[d_ind])
        print( \
            "# new data/err: {:d}/{:d}" \
                .format(len(filtered_data), err) \
        )
        ht2data_idx = collections.defaultdict(list)
        head2idx = collections.defaultdict(list)
        tail2idx = collections.defaultdict(list)
        for i, d in enumerate(tqdm(filtered_data, desc="Generating helper")):
            pair = d["entity_pair"]
            ht2data_idx[pair].append(i)
            head2idx[pair[0]].append(i)
            tail2idx[pair[1]].append(i)

        return (filtered_data, ht2data_idx, head2idx, tail2idx)
    
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
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]["sentence"]
        entity_pair = self.data[idx]["entity_pair"]
        head, tail = entity_pair
        pos_pair = (head, tail)
            
        """
        Get positive sample
        """
        pos_sample = []
        if len(self.ht2data_idx[pos_pair]) <= 2:
            pos_idx = self.ht2data_idx[pos_pair][0]
            pos_sample = [self.data[pos_idx]]
        else:
            pos_idx = random.sample( \
                self.ht2data_idx[pos_pair], 3 \
            )
            pos_sample = [ \
                self.data[pidx] \
                    for pidx in pos_idx  \
                        if self.data[pidx]["sentence"] != sentence \
            ][:1]
            if len(pos_sample) == 0:
                pos_idx = self.ht2data_idx[pos_pair][0]
                pos_sample = [self.data[pos_idx]]
        
        """
        Get hard negative samples
        """
        hard_negatives = []
        if self.config.n_hard_negs > 0:
            target_idx = self.head2idx[head] + self.tail2idx[tail]
            n_sample = self.config.n_hard_negs * 2
            if len(target_idx) < n_sample:
                target_idx *= math.ceil(n_sample / len(target_idx))
            hard_neg_idx = random.sample(target_idx, n_sample)
            hard_negatives = [ \
                self.data[neg_idx] \
                    for neg_idx in hard_neg_idx \
            ]
            hard_negatives = [ \
                d for d in hard_negatives \
                    if d["entity_pair"] != pos_pair \
            ][:self.config.n_hard_negs]

        all_sentences = \
            [sentence] \
            + [ \
                d["sentence"] \
                    for d in pos_sample \
            ] \
            + [ \
                d["sentence"] \
                    for d in hard_negatives \
            ]
        
        all_data = self.tokenize(all_sentences)
        q_data = all_data[:1]
        pos_data = all_data[1:2]
        neg_data = all_data[2:]
        
        q_data[0]["entity_pair"] = self.data[idx]["entity_pair"]
        q_data[0]["sentence"] = self.data[idx]["sentence"]
        
        pos_data[0]["entity_pair"] = pos_sample[0]["entity_pair"]
        pos_data[0]["sentence"] = pos_sample[0]["sentence"]
        
        for i in range(len(neg_data)):
            neg_data[i]["sentence"] = hard_negatives[i]["sentence"]
            neg_data[i]["entity_pair"] = hard_negatives[i]["entity_pair"]

        return {
            "input": q_data[0],
            "hard_negatives": neg_data,
            "pos_sample": pos_data
        }

def data_collator(samples):
    if len(samples) == 0:
        return {}
    bsize = len(samples) 
    inputs = []
    positives = []
    negatives = []
    for sample in samples:
        inputs.append(sample["input"])
        positives += sample["pos_sample"]
        negatives += sample["hard_negatives"]
    
    assert len(inputs) == len(positives)
    #sample, positives, negatives
    all_samples = inputs + positives + negatives
    pos_indices = list( \
        range( \
            len(inputs), \
            len(inputs) + len(positives) \
        )\
    )

    batch_mask_indices = []
    for pos_ind, inp in zip(pos_indices, inputs):
        sample_ht = inp["entity_pair"]
        mask_indices = [0] * len(all_samples)
        for inst_idx, inst in enumerate(all_samples):
            inst_ht = inst["entity_pair"]
            if inst_idx == pos_ind:
                continue
            if inst_ht == sample_ht or \
                inst["sentence"] == inp["sentence"]:
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
