import os

import apex
import collections
from dataclasses import dataclass, field
import logging
import math
import numpy as np
import json
import jsonlines
import random
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ( 
    BertTokenizer, 
    BertModel, 
    BertPreTrainedModel, 
    AdamW, 
    get_scheduler,
    get_linear_schedule_with_warmup
)
from typing import Optional

from .tokenizer import get_tokenizer
from .train_data import VKGDRDataset, data_collator
from .train_data_same_doc import VKGDRSameDocDataset, same_doc_data_collator
from .fine_tune_data import QADataset, qa_data_collator
from .techqa_pred_data import TechQAPredDataset, techqa_pred_data_collator
from .rel_encoder_data import VKGDRRelPredDataset, rel_pred_data_collator

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    init_checkpoint: Optional[str] = field(default=None)
    checkpoint_save_dir: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    eval_steps: Optional[int] = field(default=16)
    n_epochs: Optional[int] = field(default=3)
    weight_decay: Optional[float] = field(default=0.01)
    learning_rate: Optional[float] = field(default=2e-5)
    warmup_ratio: Optional[float] = field(default=0.1)

@dataclass
class RelEncoderArguments:
    init_checkpoint: Optional[str] = field(default=None)
    per_device_pred_batch_size: Optional[int] = field(default=16)
    vkg_path: Optional[str] = field(default=None)

@dataclass
class PredEncoderArguments:
    init_checkpoint: Optional[str] = field(default=None)
    per_device_pred_batch_size: Optional[int] = field(default=16)
    query_file: Optional[str] = field(default=None)

class RelationEncoderBert(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel( \
            config, \
            add_pooling_layer=False \
        )
        self.dropout = nn.Dropout( \
            config.hidden_dropout_prob)
        self.rel_encoder = nn.Linear( \
            config.hidden_size * 2, \
            config.hidden_size \
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        head_indices=None,
        tail_indices=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        bsize, length = input_ids.size()
        return_dict = self.config.use_return_dict
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        _, __, dim = sequence_output.size()

        head_vectors = torch.gather( \
            sequence_output, \
            1, \
            head_indices \
                .unsqueeze(-1) \
                .repeat(1, dim) \
                .unsqueeze(1) \
        ).squeeze(1)
        tail_vectors = torch.gather( \
            sequence_output, \
            1, \
            tail_indices \
                .unsqueeze(-1) \
                .repeat(1, dim) \
                .unsqueeze(1) \
        ).squeeze(1)
        rel_vectors = torch.cat((head_vectors, tail_vectors), dim=1)
        rel_vectors = self.rel_encoder(rel_vectors)
        return rel_vectors

class RelationEncoderTrainer:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint, \
            )
        self.model = RelationEncoderBert.from_pretrained(
            self.model_config.init_checkpoint,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.model = self.model.to(self.device)

        self.train_dataloader = None
        assert self.data_config.train_file != None
        self.train_dataset, \
            self.train_dataloader = \
                self.get_data( \
                    self.data_config.train_file, \
                    self.data_config.target_entity_pair_file, \
                    is_train=True \
                )
        self.optimizer, \
            self.lr_scheduler, \
            self.total_steps = \
            self.get_optimizer()
        
        apex.amp.register_half_function(torch, 'einsum')
        self.model, self.optimizer \
            = apex.amp.initialize(
                self.model, self.optimizer, \
                opt_level="O1")
        
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_optimizer(self):
        assert self.train_dataloader != None
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if not any(nd in n for nd in no_decay) \
                ],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if any(nd in n for nd in no_decay) \
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW( \
            optimizer_grouped_parameters, \
            lr=self.model_config.learning_rate \
        )
       
        n_steps_per_epoch = \
            math.ceil( \
                len(self.train_dataloader)\
                  / self.model_config.gradient_accumulation_steps \
            )
        train_steps = self.model_config.n_epochs \
                        * n_steps_per_epoch
        n_warmup_steps = int( \
            train_steps * self.model_config.warmup_ratio \
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=train_steps,
        )
        return (optimizer, lr_scheduler, train_steps)
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_to_save = \
            self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("\nSaving model checkpoint to {:s}".format(output_dir))
        return 0

    def get_data(self, input_file, target_entity_pair_file, is_train):
        vkgdr_dataset = VKGDRDataset( \
            input_file, \
            target_entity_pair_file, \
            self.tokenizer, \
            self.data_config \
        )
        vkgdr_dataloader = DataLoader(
            vkgdr_dataset,
            shuffle=is_train,
            collate_fn=data_collator,
            pin_memory=True,
            batch_size= \
                self.model_config.per_device_train_batch_size * self.n_gpus \
                    if is_train \
                    else self.model_config.per_device_eval_batch_size * self.n_gpus, \
            num_workers=self.n_gpus
        )
        return (vkgdr_dataset, vkgdr_dataloader)
    
    def get_loss(self, rel_vecs, pos_indices, batch_mask_indices, n_samples):
        total_n, _ = rel_vecs.size()
        input_rel_vecs, _ = \
            torch.split( \
                rel_vecs, \
                [n_samples, total_n - n_samples], \
                dim=0 \
            )
        loss_fct = nn.CrossEntropyLoss()
        rel_sim = torch.matmul( \
            input_rel_vecs, \
            rel_vecs.transpose(0, 1) \
        )
        rel_sim = rel_sim.masked_fill(batch_mask_indices.bool(), float('-inf'))
        loss = loss_fct(rel_sim, pos_indices)
        return (loss, rel_sim)

    def train_model(self):
        total_batch_size = \
            self.model_config \
                .per_device_train_batch_size \
            * self.n_gpus \
            * self.model_config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.model_config.n_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.model_config.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.model_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.total_steps}")
        
        last_checkpoint_dir = "{}_{}".format(self.model_config.checkpoint_save_dir.rstrip("/"), "last_checkpoint")
        total_train_loss = 0.0
        best_val_acc = float("-inf")
        global_steps = 0
        self.model.zero_grad()
        for i in range(0, self.model_config.n_epochs):
            self.model.train()
            progress_bar = tqdm( \
                self.train_dataloader, \
                desc="Training ({:d}'th iter / {:d})".format(i+1, self.model_config.n_epochs, 0.0, 0.0) \
            )
            for step, batch in enumerate(progress_bar):
                self.model.train()
                target_inputs = [ \
                    "input_ids", "token_type_ids", \
                    "attention_mask", "head_indices", \
                    "tail_indices" \
                ]
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                rel_vecs = self.model(**new_batch)
                output = self.get_loss( \
                    rel_vecs, \
                    batch["pos_indices"].to(self.device), \
                    batch["batch_mask_indices"].to(self.device), \
                    batch["n_samples"]
                )
                loss = output[0]
                loss = loss / self.model_config.gradient_accumulation_steps
                
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                
                if step % self.model_config.gradient_accumulation_steps == self.model_config.gradient_accumulation_steps - 1 \
                    or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_steps += 1
                total_train_loss += loss.item()
                
                if global_steps % self.model_config.eval_steps == 1:
                    avg_train_loss = \
                        total_train_loss \
                            / self.model_config.eval_steps
                    self.save_model(last_checkpoint_dir)
                    desc_template = "Training ({:d}'th iter / {:d}) | loss: {:.03f}"
                    print('\n')
                    print(desc_template.format( \
                            i+1, \
                            self.model_config.n_epochs, \
                            avg_train_loss, \
                        )
                    )
                    total_train_loss = 0.0
        self.save_model(last_checkpoint_dir)
        return 0

class RelationEncoderTrainerSameDoc:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint, \
            )
        self.model = RelationEncoderBert.from_pretrained(
            self.model_config.init_checkpoint,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.model = self.model.to(self.device)

        self.train_dataloader = None
        self.eval_dataloader = None
        assert self.data_config.train_file != None \
            and self.data_config.eval_file != None
        self.train_dataset, \
            self.train_dataloader = \
                self.get_data( \
                    self.data_config.train_file, \
                    self.data_config.target_entity_pair_file, \
                    is_train=True \
                )
        self.eval_dataset, \
            self.eval_dataloader = \
                self.get_data( \
                    self.data_config.eval_file, \
                    self.data_config.target_entity_pair_file, \
                    is_train=False \
                )
        self.optimizer, \
            self.lr_scheduler, \
            self.total_steps = \
            self.get_optimizer()
        
        apex.amp.register_half_function(torch, 'einsum')
        self.model, self.optimizer \
            = apex.amp.initialize(
                self.model, self.optimizer, \
                opt_level="O1")
        
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_optimizer(self):
        assert self.train_dataloader != None
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if not any(nd in n for nd in no_decay) \
                ],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if any(nd in n for nd in no_decay) \
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW( \
            optimizer_grouped_parameters, \
            lr=self.model_config.learning_rate \
        )
       
        n_steps_per_epoch = \
            math.ceil( \
                len(self.train_dataloader)\
                  / self.model_config.gradient_accumulation_steps \
            )
        train_steps = self.model_config.n_epochs \
                        * n_steps_per_epoch
        n_warmup_steps = int( \
            train_steps * self.model_config.warmup_ratio \
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=train_steps,
        )
        return (optimizer, lr_scheduler, train_steps)
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_to_save = \
            self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("\nSaving model checkpoint to {:s}".format(output_dir))
        return 0

    def get_data(self, input_file, target_entity_pair_file, is_train):
        vkgdr_dataset = VKGDRSameDocDataset( \
            input_file, \
            target_entity_pair_file, \
            self.tokenizer, \
            self.data_config \
        )
        vkgdr_dataloader = DataLoader(
            vkgdr_dataset,
            shuffle=is_train,
            collate_fn=same_doc_data_collator,
            pin_memory=True,
            batch_size= \
                self.model_config.per_device_train_batch_size * self.n_gpus \
                    if is_train \
                    else self.model_config.per_device_eval_batch_size * self.n_gpus, \
            num_workers=self.n_gpus
        )
        return (vkgdr_dataset, vkgdr_dataloader)
    
    def get_loss(self, rel_vecs, pos_indices, batch_mask_indicies, n_samples):
        total_n, _ = rel_vecs.size()
        input_rel_vecs, _ = \
            torch.split( \
                rel_vecs, \
                [n_samples, total_n - n_samples], \
                dim=0 \
            )
        loss_fct = nn.CrossEntropyLoss()
        rel_sim = torch.matmul( \
            input_rel_vecs, \
            rel_vecs.transpose(0, 1) \
        )
        rel_sim = rel_sim.masked_fill( \
            batch_mask_indicies.bool(), \
            float('-inf') \
        )
        loss = loss_fct(rel_sim, pos_indices)
        return (loss, rel_sim)

    def train_model(self):
        total_batch_size = \
            self.model_config \
                .per_device_train_batch_size \
            * self.n_gpus \
            * self.model_config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.model_config.n_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.model_config.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.model_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.total_steps}")
        
        last_checkpoint_dir = "{}_{}".format(self.model_config.checkpoint_save_dir.rstrip("/"), "last_checkpoint")
        total_train_loss = 0.0
        best_val_acc = float("-inf")
        global_steps = 0
        self.model.zero_grad()
        for i in range(0, self.model_config.n_epochs):
            self.model.train()
            progress_bar = tqdm( \
                self.train_dataloader, \
                desc="Training ({:d}'th iter / {:d})".format(i+1, self.model_config.n_epochs, 0.0, 0.0) \
            )
            for step, batch in enumerate(progress_bar):
                self.model.train()
                target_inputs = [ \
                    "input_ids", "token_type_ids", \
                    "attention_mask", "head_indices", \
                    "tail_indices" \
                ]
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                rel_vecs = self.model(**new_batch)
                output = self.get_loss( \
                    rel_vecs, \
                    batch["pos_indices"].to(self.device), \
                    batch["batch_mask_indices"].to(self.device), \
                    batch["n_samples"]
                )
                loss = output[0]
                loss = loss / self.model_config.gradient_accumulation_steps
                
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                
                if step % self.model_config.gradient_accumulation_steps == self.model_config.gradient_accumulation_steps - 1 \
                    or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_steps += 1
                total_train_loss += loss.item()
                
                if global_steps % self.model_config.eval_steps == 1:
                    avg_train_loss = \
                        total_train_loss \
                            / self.model_config.eval_steps
                    val_acc = self.eval_model()
                    if val_acc > best_val_acc:
                        self.save_model(self.model_config.checkpoint_save_dir)
                        best_val_acc = val_acc
                    self.save_model(last_checkpoint_dir)
                    desc_template = "Training ({:d}'th iter / {:d})|loss: {:.03f}, Val: {:.03f}"
                    print('\n')
                    print(desc_template.format( \
                            i+1, \
                            self.model_config.n_epochs, \
                            avg_train_loss, \
                            val_acc \
                        )
                    )
                    total_train_loss = 0.0
        val_acc = self.eval_model()
        if val_acc > best_val_acc:
            self.save_model(self.model_config.checkpoint_save_dir)
            best_val_acc = val_acc
        self.save_model(last_checkpoint_dir)
        desc_template = "loss: {:.03f}, Val: {:.03f}"
        print('\n')
        print(desc_template.format( \
                avg_train_loss, \
                val_acc \
            )
        )
        return 0
    
    def eval_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", \
            "attention_mask", "head_indices", \
            "tail_indices" \
        ]
        self.model.eval()
        predictions = []
        labels = []
        for batch in tqdm(self.eval_dataloader, desc="Eval"):
            with torch.no_grad():
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                rel_vecs = self.model(**new_batch)
                output = self.get_loss( \
                    rel_vecs, \
                    batch["pos_indices"].to(self.device), \
                    batch["batch_mask_indices"].to(self.device), \
                    batch["n_samples"]
                )
                rel_sim = output[1].detach().cpu().numpy()
                positive_indices = batch["pos_indices"].detach().cpu().numpy()
            predictions.append(np.argmax(rel_sim, axis=1))
            labels.append(positive_indices)
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        # Calc acc
        acc = np.mean(predictions == labels)
        return acc

class RelationEncoderFineTuner:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint, \
            )
        self.model = RelationEncoderBert.from_pretrained(
            self.model_config.init_checkpoint,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.model = self.model.to(self.device)

        self.train_dataloader = None
        self.eval_dataloader = None
        assert self.data_config.train_file != None \
            and self.data_config.eval_file != None
        self.train_dataset, \
            self.train_dataloader = \
                self.get_data( \
                    self.data_config.train_file, \
                    self.data_config.corpus_file, \
                    self.data_config.target_entity_pair_file, \
                    is_train=True \
                )
        self.eval_dataset, \
            self.eval_dataloader = \
                self.get_data( \
                    self.data_config.eval_file, \
                    self.data_config.corpus_file, \
                    self.data_config.target_entity_pair_file, \
                    is_train=False \
                )
        self.optimizer, \
            self.lr_scheduler, \
            self.total_steps = \
            self.get_optimizer()
        
        apex.amp.register_half_function(torch, 'einsum')
        self.model, self.optimizer \
            = apex.amp.initialize(
                self.model, self.optimizer, \
                opt_level="O1")
        
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_optimizer(self):
        assert self.train_dataloader != None
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if not any(nd in n for nd in no_decay) \
                ],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if any(nd in n for nd in no_decay) \
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW( \
            optimizer_grouped_parameters, \
            lr=self.model_config.learning_rate \
        )
       
        n_steps_per_epoch = \
            math.ceil( \
                len(self.train_dataloader)\
                  / self.model_config.gradient_accumulation_steps \
            )
        train_steps = self.model_config.n_epochs \
                        * n_steps_per_epoch
        n_warmup_steps = int( \
            train_steps * self.model_config.warmup_ratio \
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=train_steps,
        )
        return (optimizer, lr_scheduler, train_steps)
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_to_save = \
            self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("\nSaving model checkpoint to {:s}".format(output_dir))
        return 0

    def get_data(self, input_file, corpus_file, target_entity_pair_file, is_train):
        vkgdr_dataset = QADataset( \
            input_file, \
            corpus_file, \
            target_entity_pair_file, \
            self.tokenizer, \
            self.data_config \
        )
        vkgdr_dataloader = DataLoader(
            vkgdr_dataset,
            shuffle=is_train,
            collate_fn=qa_data_collator,
            pin_memory=True,
            batch_size= \
                self.model_config.per_device_train_batch_size * self.n_gpus \
                    if is_train \
                    else self.model_config.per_device_eval_batch_size * self.n_gpus, \
            num_workers=self.n_gpus
        )
        return (vkgdr_dataset, vkgdr_dataloader)
    
    def get_loss(self, rel_vecs, pos_indices, batch_mask_indicies, n_samples):
        total_n, _ = rel_vecs.size()
        input_rel_vecs, _ = \
            torch.split( \
                rel_vecs, \
                [n_samples, total_n - n_samples], \
                dim=0 \
            )
        loss_fct = nn.CrossEntropyLoss()
        rel_sim = torch.matmul( \
            input_rel_vecs, \
            rel_vecs.transpose(0, 1) \
        )
        rel_sim = rel_sim.masked_fill( \
            batch_mask_indicies.bool(), \
            float('-inf') \
        )
        loss = loss_fct(rel_sim, pos_indices)
        return (loss, rel_sim)

    def train_model(self):
        total_batch_size = \
            self.model_config \
                .per_device_train_batch_size \
            * self.n_gpus \
            * self.model_config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.model_config.n_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.model_config.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.model_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.total_steps}")
        
        last_checkpoint_dir = "{}_{}".format(self.model_config.checkpoint_save_dir.rstrip("/"), "last_checkpoint")
        total_train_loss = 0.0
        best_val_acc = float("-inf")
        global_steps = 0
        self.model.zero_grad()
        for i in range(0, self.model_config.n_epochs):
            self.model.train()
            progress_bar = tqdm( \
                self.train_dataloader, \
                desc="Training ({:d}'th iter / {:d})".format(i+1, self.model_config.n_epochs, 0.0, 0.0) \
            )
            for step, batch in enumerate(progress_bar):
                self.model.train()
                target_inputs = [ \
                    "input_ids", "token_type_ids", \
                    "attention_mask", "head_indices", \
                    "tail_indices" \
                ]
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                rel_vecs = self.model(**new_batch)
                output = self.get_loss( \
                    rel_vecs, \
                    batch["pos_indices"].to(self.device), \
                    batch["batch_mask_indices"].to(self.device), \
                    batch["n_samples"]
                )
                loss = output[0]
                loss = loss / self.model_config.gradient_accumulation_steps
                
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                
                if step % self.model_config.gradient_accumulation_steps == self.model_config.gradient_accumulation_steps - 1 \
                    or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_steps += 1
                total_train_loss += loss.item()
                
                if global_steps % self.model_config.eval_steps == 1:
                    avg_train_loss = \
                        total_train_loss \
                            / self.model_config.eval_steps
                    val_acc = self.eval_model()
                    if val_acc > best_val_acc:
                        self.save_model(self.model_config.checkpoint_save_dir)
                        best_val_acc = val_acc
                    self.save_model(last_checkpoint_dir)
                    desc_template = "Training ({:d}'th iter / {:d})|loss: {:.03f}, Val: {:.03f}"
                    print('\n')
                    print(desc_template.format( \
                            i+1, \
                            self.model_config.n_epochs, \
                            avg_train_loss, \
                            val_acc \
                        )
                    )
                    total_train_loss = 0.0
        val_acc = self.eval_model()
        if val_acc > best_val_acc:
            self.save_model(self.model_config.checkpoint_save_dir)
            best_val_acc = val_acc
        self.save_model(last_checkpoint_dir)
        desc_template = "loss: {:.03f}, Val: {:.03f}"
        print('\n')
        print(desc_template.format( \
                avg_train_loss, \
                val_acc \
            )
        )
        return 0
    
    def eval_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", \
            "attention_mask", "head_indices", \
            "tail_indices" \
        ]
        self.model.eval()
        predictions = []
        labels = []
        for batch in tqdm(self.eval_dataloader, desc="Eval"):
            with torch.no_grad():
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                rel_vecs = self.model(**new_batch)
                output = self.get_loss( \
                    rel_vecs, \
                    batch["pos_indices"].to(self.device), \
                    batch["batch_mask_indices"].to(self.device), \
                    batch["n_samples"]
                )
                rel_sim = output[1].detach().cpu().numpy()
                positive_indices = batch["pos_indices"].detach().cpu().numpy()
            predictions.append(np.argmax(rel_sim, axis=1))
            labels.append(positive_indices)
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        # Calc acc
        acc = np.mean(predictions == labels)
        return acc

class VKGConstructor:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint, \
            )
        self.model = RelationEncoderBert.from_pretrained(
            self.model_config.init_checkpoint,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.model = self.model.to(self.device) 

        assert self.data_config.corpus_file != None
        self.pred_dataset, \
            self.pred_dataloader = \
                self.get_data( \
                    self.data_config.corpus_file, \
                    self.data_config.target_pair_file \
                )

        apex.amp.register_half_function(torch, 'einsum')
        self.model = apex.amp.initialize(
            self.model, \
            opt_level="O1" \
        )
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_data(self, input_file, target_pair_file):
        rel_pred_dataset = VKGDRRelPredDataset( \
            input_file, \
            target_pair_file, \
            self.tokenizer, \
            self.data_config \
        )
        rel_pred_dataloader = DataLoader(
            rel_pred_dataset,
            shuffle=False,
            pin_memory=True,
            collate_fn=rel_pred_data_collator,
            batch_size= \
                self.model_config.per_device_pred_batch_size * self.n_gpus, \
            num_workers=self.n_gpus
        )
        return (rel_pred_dataset, rel_pred_dataloader)

    def pred_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", \
            "attention_mask", "head_indices", \
            "tail_indices" \
        ]
        self.model.eval()
        np_file = os.path.join(self.model_config.vkg_path, "rel_vec.npy")
        ht_file = os.path.join(self.model_config.vkg_path, "ht.jsonl")
        n_file = os.path.join(self.model_config.vkg_path, "n_batches.json")
        n_batches = 0
        
        os.makedirs(self.model_config.vkg_path, exist_ok=True) 
        with jsonlines.open(ht_file, "w") as writer:
            with open(np_file, "wb") as fw:
                for batch in tqdm(self.pred_dataloader, desc="Eval"):
                    if len(batch) == 0:
                        continue
                    with torch.no_grad():
                        new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                        batch_rel_vecs = self.model(**new_batch)
                        batch_rel_vecs = batch_rel_vecs.detach().cpu().numpy().astype(np.float16)
                        
                        doc_ids = batch["doc_ids"]
                        entity_pairs = batch["entity_pairs"]
                        for doc_id, entity_pair in zip(doc_ids, entity_pairs):
                            h, t = entity_pair
                            writer.write({
                                "head": h, 
                                "tail": t,
                                "doc_id": doc_id
                            })
                        np.save(fw, batch_rel_vecs)
                    n_batches += 1
        with open(n_file, "w") as f:
            json.dump(n_batches, f)
        return 0

class DocRetriever:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint, \
            )
        self.model = RelationEncoderBert.from_pretrained(
            self.model_config.init_checkpoint,
            output_attentions=False,
            output_hidden_states=True,
        )
        self.model = self.model.to(self.device) 

        apex.amp.register_half_function(torch, 'einsum')
        self.model = apex.amp.initialize(
            self.model, \
            opt_level="O1" \
        )
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()

    def get_data(self, input_file):
        raise Exception("Not implemented yet")

    def compute_query_vectors(self, pred_dataloader):
        target_inputs = [ \
            "input_ids", "token_type_ids", \
            "attention_mask", "head_indices", \
            "tail_indices" \
        ]

        qid2queries = {}
        for batch in tqdm(pred_dataloader, desc="Pred"):
            with torch.no_grad():
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                rel_vecs = self.model(**new_batch)
                rel_vecs = rel_vecs.detach().cpu().numpy()
                
                qids = batch["qids"]
                head_tails = batch["head_and_tails"]
                gt_doc_ids = batch["gt_docs"]
                candidate_doc_ids = batch["candidate_doc_ids"]
                
                for qid, head_tail, v, gt_doc_id, candidate_docs in zip( \
                    qids, head_tails, rel_vecs, gt_doc_ids, candidate_doc_ids \
                ):
                    if qid not in qid2queries:
                        qid2queries[qid] = {
                            "qid": qid,
                            "head_tail": [],
                            "qvec": [],
                            "gt_doc_id": None,
                            "candidate_doc_ids": None,
                        }
                    qid2queries[qid]["head_tail"].append(head_tail)
                    qid2queries[qid]["qvec"].append(v)
                    qid2queries[qid]["gt_doc_id"] = gt_doc_id
                    qid2queries[qid]["candidate_doc_ids"] = candidate_docs
        
        for qid in qid2queries:
            qid2queries[qid]["qvec"] = np.stack(qid2queries[qid]["qvec"], axis=0)
        return qid2queries

    def pred_single_hop(self, pred_dataloader):
        assert self.model_config.query_file != None
        qid2queries = self.compute_query_vectors(pred_dataloader)
        with jsonlines.open(self.model_config.query_file, "w") as writer:
            for qid in tqdm(qid2queries, desc="Querying"):
                q = qid2queries[qid]
                result = { 
                    "qid": q["qid"],
                    "gt": q["gt_doc_id"],
                    "qvec": q["qvec"].tolist(),
                    "q_entity_pair": q["head_tail"],
                }
                writer.write(result)
        return 0
    
    def pred_model(self):
        pred_dataset, pred_dataloader = \
            self.get_data( \
                self.data_config.pred_file
            )
        _ = self.pred_single_hop(pred_dataloader)
        return 0

class TechQADocRetriever(DocRetriever):
    def __init__(self, model_config, data_config):
        super().__init__(model_config, data_config)

    def get_data(self, input_file):
        pred_dataset = TechQAPredDataset( \
            input_file, \
            self.tokenizer, \
            self.data_config \
        )
        pred_dataloader = DataLoader(
            pred_dataset,
            shuffle=False,
            pin_memory=True,
            collate_fn=techqa_pred_data_collator,
            batch_size= \
                self.model_config.per_device_pred_batch_size * self.n_gpus, \
            num_workers=self.n_gpus
        )
        return (pred_dataset, pred_dataloader)
