import collections
from collections import namedtuple
import json
import jsonlines
import numpy as np
import os
import random
import string
from tqdm import tqdm
import urllib.parse

from .wikipedia_preprocessor import WikipediaPreprocessor
from .wikidata_utils import WikiDataParser

class PTDGenerator:
    def __init__(self, args):
        self.args = args

        self.funcs = []
        if args.preprocess_wikipedia:
            self.funcs.append(self.preprocess_wikipedia_and_save)
        if args.get_wikiid2wikidata:
            self.funcs.append(self.get_wikiid2wikidataid)
        if args.get_entities:
            self.funcs.append(self.get_entities)
        if args.build_pretraining_data:
            self.funcs.append(self.build_pretraining_data)
        if args.filter_sentences_and_get_target_pairs:
            self.funcs.append(self.filter_sentences_and_get_target_pairs)
        if args.split_train_valid:
            self.funcs.append(self.split_train_valid)

    def run(self):
        for func in self.funcs:
            func()
        return 0

    def preprocess_wikipedia_and_save(self):
        assert os.path.isdir(self.args.wikipedia_dump_file) \
            and self.args.wiki_prep_file != None
        WikiProcessArgs = namedtuple( \
            "WikiProcessArgs", \
            ["wikidump_file", "wiki_prep_file"] \
        )
        args = WikiProcessArgs( \
            wikidump_file=self.args.wikipedia_dump_file, \
            wiki_prep_file=self.args.wiki_prep_file \
        )
        wiki_prep = WikipediaPreprocessor(args)
        _ = wiki_prep.run()
        return 0

    def get_wikiid2wikidataid(self):
        assert os.path.isfile(self.args.wikidata_dump_file) \
            and self.args.wikiid2wikidataid_file != None
        WikiDataProcessArgs = namedtuple( \
            "WikiDataProcessArgs", \
            ["wikidata_dump_file"] \
        )
        args = WikiDataProcessArgs( \
            wikidata_dump_file=self.args.wikidata_dump_file, \
        )
        wikidata_parser = WikiDataParser(args)
        results = wikidata_parser.construct_wikiid2wikidataid()
        with open(self.args.wikiid2wikidataid_file, "w") as f:
            json.dump(results, f)
        return 0
     
     #Code reference: https://github.com/wenhuchen/KGPT/blob/main/preprocess/step2.py
    def _to_wikilink(self, text):
        modified = [_.capitalize() for _ in text.split(' ')]
        modified = '_'.join(modified)
        return modified

    def _parse_wiki_hyperlink(self, hyperlink):
        modified = urllib.parse.unquote(hyperlink)
        modified = self._to_wikilink(modified)
        return modified

    def _parse_wikipedia_title(self, title):
        return self._to_wikilink(title)   
    
    def get_entities(self):
        assert os.path.isfile(self.args.wiki_prep_file) \
                and os.path.isfile(self.args.wikiid2wikidataid_file) \
                and self.args.wiki_entity_file != None
        
        with open(self.args.wikiid2wikidataid_file, "r") as f:
            wikiid2wikidataid = json.load(f)
        
        err = 0
        with jsonlines.open(self.args.wiki_entity_file, "w") as writer:
            with jsonlines.open(self.args.wiki_prep_file, "r") as reader:
                progress_bar = tqdm(reader, desc="Getting Entities")
                for n, wiki_page in enumerate(progress_bar):
                    title = wiki_page["title"]
                    wikiid = wiki_page["id"]
                    if wikiid not in wikiid2wikidataid:
                        err += 1
                        continue
                    wikidataid = wikiid2wikidataid[wikiid]
                    wikilink = self._parse_wikipedia_title(title)
                    writer.write({
                        "entity": title,
                        "kg_id": wikidataid,
                        "wikilink": wikilink,
                    })
                    if n % 1000 == 0:
                        progress_bar.set_description(
                            "Missing entities: {:d}, {:d}, {:.04f}" \
                                .format( \
                                    err, n, err * 1.0 / (n + 1) \
                                ) \
                        )
        return 0

    def build_pretraining_data(self):
        assert os.path.isfile(self.args.wiki_prep_file) \
                and os.path.isfile(self.args.wiki_entity_file)
        assert self.args.wikipedia_sentences_file != None
        
        with jsonlines.open(self.args.wiki_entity_file, "r") as reader:
            wikilink2wikidataid = {}
            for inst in tqdm(reader, desc="Building wikilink2wikidataid"):
                wikilink2wikidataid[inst["wikilink"]] = inst["kg_id"]
        
        with jsonlines.open(self.args.wikipedia_sentences_file, "w") as writer:
            with jsonlines.open(self.args.wiki_prep_file, "r") as reader:
                n, err = (0, 0)
                progress_bar = tqdm(reader, desc="Generating pretraining data")
                for page_i, wiki_page in enumerate(progress_bar):
                    if page_i % 1000 == 0 and n > 0:
                        progress_bar.set_description("Generating pretraining data | Err {}/{}: {:.04f}".format(err, n, err/n))
                    for sent in wiki_page["text"]:
                        tokens = []
                        entities = []
                        for i, token in enumerate(sent):
                            t, h = token
                            tokens.append(t)
                            if h != None:
                                wikilink = self._parse_wiki_hyperlink(h)
                                if wikilink not in wikilink2wikidataid:
                                    err += 1
                                else:
                                    entities.append( \
                                      (i, wikilink2wikidataid[wikilink]) \
                                    )
                                n += 1
                        if len(entities) < 2:
                            continue
                        writer.write({
                            "sentence": tokens,
                            "entities": entities, \
                            "target_entity_pairs": "all"
                        })
        print("Error cases: {:.04f}%({}/{})".format(100 * err/n, err, n))
        return 0

    def filter_sentences_and_get_target_pairs(self):
        assert self.args.wikipedia_sentences_file != None \
            and self.args.target_entity_pair_file != None \
            and self.args.entity_pair_n_thresh != None \
            and self.args.n_filter != None
        
        n_sent = 0 
        entity2n = collections.defaultdict(int)
        pair2n = collections.defaultdict(int)
        with jsonlines.open(self.args.wikipedia_sentences_file, "r") as reader:
            for sent in tqdm(reader, desc="Filtering (stat)"):
                n_sent += 1
                sentence = sent["sentence"]
                entities = sent["entities"]
                eids = set([eid for _, eid in entities])
                for entity_id in eids:
                    entity2n[entity_id] += 1
                pairs = []
                if sent["target_entity_pairs"] == "all":
                    pairs = set([ \
                        (ent_id_1, ent_id_2) \
                            for idx_1, ent_id_1 in entities \
                                for idx_2, ent_id_2 in entities \
                                    if ent_id_1 != ent_id_2 \
                    ])
                else:
                    pairs = set([ \
                        (entities[i][1], entities[j][1]) \
                            for i, j in sent["target_entity_pairs"]
                    ])
                for pair in pairs:
                    pair2n[pair] += 1
        
        #Filter by occ
        filtered_pairs = set([ \
            pair for pair in tqdm(pair2n, desc="Filtering (by N)") if pair2n[pair] >= self.args.n_filter \
        ])
        #Get pmi
        pair2pmi = {}
        for pair in tqdm(filtered_pairs, desc="Filtering (Cal PMI)"):
            e1, e2 = pair
            pe1 = entity2n[e1] / n_sent
            pe2 = entity2n[e2] / n_sent
            p_e1_e2 = pair2n[pair] / n_sent
            pair2pmi[pair] = \
                np.log(p_e1_e2) - np.log(pe1) - np.log(pe2)
        
        #Filter by PMI
        pair_pmi = [(pair, pair2pmi[pair]) for pair in tqdm(filtered_pairs, desc="Filtering (PMI processing)")]
        print("Filtering (by PMI)")
        pair_pmi.sort(key=lambda x: x[1], reverse=True)
        pair_pmi = pair_pmi[:self.args.entity_pair_n_thresh]
        with jsonlines.open(self.args.target_entity_pair_file, "w") as writer:
            for pair, pmi in pair_pmi:
                writer.write({
                    "head": pair[0],
                    "tail": pair[1],
                })
        return 0

    def split_train_valid(self):
        assert self.args.pretraining_data_train_file != None \
            and self.args.pretraining_data_valid_file != None \
            and self.args.wikipedia_sentences_file != None \
            and self.args.train_prop != None
        
        with jsonlines.open(self.args.pretraining_data_train_file, "w") as tr_writer:
            with jsonlines.open(self.args.pretraining_data_valid_file, "w") as va_writer:
                with jsonlines.open(self.args.wikipedia_sentences_file, "r") as reader:
                    for d in tqdm(reader, desc="Spliting the PreTraining file"):
                        if np.random.rand() < self.args.train_prop:
                            tr_writer.write(d)
                        else:
                            va_writer.write(d)
        return 0
