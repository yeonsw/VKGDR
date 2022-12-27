#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
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
class PredDataArguments:
    max_length: Optional[int] = field(default=128)
    pred_file: Optional[str] = field(default=None)

class TechQAPredDataset(Dataset):
    def __init__(self, input_file, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length \
            = self.config.max_length
        self.data = self.get_data(input_file)
    
    def get_masked_sentences_from_sentence(self, inp):
        tokens = inp["question_tokens"] # List of tokens
        candidate_doc_ids = inp["candidate_doc_ids"]
        doc_id = inp["doc_id"]
        qid = inp["qid"]
        target_entity_pairs = inp["target_entity_pairs"]
        mentions = inp["mentions"]
        
        head_and_tails = []
        masked_sentences = []
        for hind, tind in target_entity_pairs:
            _, hid = mentions[hind]
            _, tid = mentions[tind]
            
            head_and_tails.append((hid, tid))

            masked = tokens[:]
            pair = [ \
                (hind, "[unused1]"), (tind, "[unused2]")
            ]
            pair.sort( \
                key=lambda x: x[0], \
                reverse=True \
            )
            for target_ind, sp_token in pair:
                masked.insert( \
                    target_ind + 1, sp_token \
                )
                masked[target_ind] = "[unused3]"
            masked = " ".join(masked)
            masked_sentences.append(masked)
        if len(masked_sentences) == 0:
            masked = tokens[:]
            masked = tokens \
                + ["[unused3]"] + ["[unused1]"] \
                + ["[unused3]"] + ["[unused2]"]
            masked_sentences.append(" ".join(masked))
            head_and_tails.append(("0", "1"))
        return {
            "question_tokens": tokens,
            "masked_questions": masked_sentences,
            "head_and_tails": head_and_tails,
            "doc_id": doc_id,
            "candidate_doc_ids": candidate_doc_ids,
            "qid": qid
        }
    
    def get_data(self, fname):
        with jsonlines.open(fname, "r") as reader:
            sentences = [r for r in reader]
        
        data = [] 
        for sent in tqdm(sentences, desc="Preprocessing sentences"):
            data.append(self.get_masked_sentences_from_sentence(sent))
        
        r1_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused1]')
        r2_token = \
            self.tokenizer \
                .convert_tokens_to_ids('[unused2]')
        
        err = 0
        qids = []
        new_data = []
        progress_bar = tqdm(range(len(data)), desc="Generating new data")
        for input_idx in progress_bar:
            preprocessed_questions = self.tokenizer.batch_encode_plus( \
                data[input_idx]["masked_questions"], \
                padding="max_length", \
                max_length=self.max_length, \
                truncation=True, \
                return_tensors='pt' \
            )
            head_tail_ids = data[input_idx]["head_and_tails"]
            for ind in range(len(preprocessed_questions["input_ids"])):
                input_ids = preprocessed_questions["input_ids"][ind]
                r1_idx = torch.nonzero(
                    (input_ids == r1_token)
                ).flatten().tolist()
                r2_idx = torch.nonzero(
                    (input_ids == r2_token)
                ).flatten().tolist()
                if len(r1_idx) == 0 \
                    or len(r2_idx) == 0:
                    err += 1
                    continue
                qids.append(data[input_idx]["qid"])
                new_data.append({
                    "qid": data[input_idx]["qid"],
                    "question_tokens": data[input_idx]["question_tokens"],
                    "gt_doc": data[input_idx]["doc_id"],
                    "candidate_doc_ids": data[input_idx]["candidate_doc_ids"],
                    "head_idx": r1_idx[0],
                    "tail_idx": r2_idx[0],
                    "head_tail_ids": head_tail_ids[ind]
                })
                features = list(preprocessed_questions.keys())
                for key in features:
                    new_data[-1][key] = preprocessed_questions[key][ind]
            if input_idx % 2 == 0:
                progress_bar.set_description( \
                    "Generating new data / Err ({:d}/{:d})" \
                        .format(err, len(new_data)) \
                )
        print( \
            "Generating new data / Err ({:d}/{:d})" \
                .format(err, len(data)) \
        )
        print("N Qids: {}".format(len(set(qids))))
        return new_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def techqa_pred_data_collator(samples):
    if len(samples) == 0:
        return {}
    bsize = len(samples)
    inputs = samples
    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    head_indices = []
    tail_indices = []
    for inp in inputs:
        input_ids.append( \
            inp["input_ids"] \
        )
        token_type_ids.append( \
            inp["token_type_ids"] if "token_type_ids" in inp else None\
        )
        attention_mask.append( \
            inp["attention_mask"] \
        )
        head_indices.append( \
            inp["head_idx"]
        )
        tail_indices.append( \
            inp["tail_idx"]
        )
    input_ids = torch.stack(input_ids, dim=0)
    if token_type_ids[0] != None:
        token_type_ids = torch.stack( \
            token_type_ids, dim=0 \
        )
    else:
        token_type_ids = None
    attention_mask = torch.stack( \
        attention_mask, dim=0 \
    )
    head_indices = torch.tensor( \
        head_indices, dtype=torch.long \
    )
    tail_indices = torch.tensor( \
        tail_indices, dtype=torch.long \
    )
    original_questions = [s["question_tokens"] for s in samples]
    qids = [s["qid"] for s in samples]
    gt_docs = [s["gt_doc"] for s in samples]
    head_and_tails = [s["head_tail_ids"] for s in samples]
    candidate_doc_ids = [s["candidate_doc_ids"] for s in samples]
    new_batch = {
        "n_samples": bsize,
        "qids": qids,
        "original_questions": original_questions,
        "gt_docs": gt_docs,
        "candidate_doc_ids": candidate_doc_ids,
        "head_and_tails": head_and_tails,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "head_indices": head_indices,
        "tail_indices": tail_indices,
    }
    if token_type_ids != None:
        new_batch["token_type_ids"] = token_type_ids
    return new_batch
