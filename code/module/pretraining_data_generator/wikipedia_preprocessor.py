#Code reference: https://github.com/wenhuchen/KGPT/tree/main/preprocess
import argparse
import bz2
import json
import jsonlines
from tqdm import tqdm
import os

class WikipediaPreprocessor:
    def __init__(self, args):
        self.args = args
    
    def get_all_files_in_dir(self, wiki_dir):
        wiki_paths = [ \
            os.path.join(wiki_dir, path) \
                for path in os.listdir(wiki_dir) \
                    if os.path.isdir(os.path.join(wiki_dir, path)) \
        ]
        wiki_paths.sort()
        
        wiki_files = []
        for path in wiki_paths:
            wiki_files += [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(".bz2")]
        wiki_files.sort()
        print("N Wikipedia files: {:d}".format(len(wiki_files)))
        return wiki_files

    def run(self):
        wiki_files = self.get_all_files_in_dir( \
            self.args.wikidump_file)
        n_pages = 0
        os.makedirs(os.path.dirname(self.args.wiki_prep_file), exist_ok=True) 
        with jsonlines.open(self.args.wiki_prep_file, 'w') as fw:
            tqdm_wiki_files = tqdm(wiki_files, desc="Preprocessing {:d} pages".format(n_pages))
            for wiki_file in tqdm_wiki_files:
                with bz2.BZ2File(wiki_file, "r") as fr:
                    pages = fr.readlines()
                    n_pages += len(pages)
                    tqdm_wiki_files.set_description("Preprocessing {:d} pages".format(n_pages))
                    for i, line in enumerate(pages):
                        data = json.loads(line)
                        outputs = self.preprocess_wikipedia_page(data)
                        if len(outputs) > 0:
                            entry = { 
                                "id": data['id'], 
                                'title': data['title'], 
                                'text': outputs 
                            }
                            fw.write(entry)
        return 0

    def preprocess_wikipedia_page(self, data):
        outputs = []
        paragraphs = data["text"][1:] # ignore title
        char_offsets = data["charoffset"][1:] # ignore title
        for p, offset in zip(paragraphs, char_offsets):
            paragraph = "".join(p)
            for sent_offsets in offset:
                tokens = []
                sentence_tokens = [paragraph[s:e] for s, e in sent_offsets]
                
                hyperlink, target_token = (None, [])
                stacking_target_token = False
                for i, token in enumerate(sentence_tokens):
                    if (not stacking_target_token) and '<a href="' in token: 
                        stoken = '<a href="'
                        etoken = '">'
                        hyperlink = token[len(stoken):-len(etoken)]
                        stacking_target_token = True
                        continue
                    
                    if stacking_target_token and token == '</a>':
                        stacking_target_token = False
                        tokens.append((" ".join(target_token), hyperlink))
                        hyperlink=None
                        target_token = []
                        continue
                    
                    if stacking_target_token:
                        target_token.append(token)
                        continue
                    tokens.append((token, hyperlink))
                if len(tokens) < 8:
                    continue
                outputs.append(tokens)
        return outputs

