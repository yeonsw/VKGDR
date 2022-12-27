import argparse

from module.pretraining_data_generator import data_builder

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preprocess_wikipedia", action="store_true")
    parser.add_argument("--get_wikiid2wikidata", action="store_true")
    parser.add_argument("--get_entities", action="store_true")
    parser.add_argument("--build_pretraining_data", action="store_true")
    parser.add_argument("--filter_sentences_and_get_target_pairs", action="store_true")
    parser.add_argument("--split_train_valid", action="store_true")
    
    parser.add_argument("--wikipedia_dump_file", type=str, default=None)
    parser.add_argument("--wiki_prep_file", type=str, default=None)
    parser.add_argument("--wikidata_dump_file", type=str, default=None)
    parser.add_argument("--wikiid2wikidataid_file", type=str, default=None)
    parser.add_argument("--wiki_entity_file", type=str, default=None)
    parser.add_argument("--wikipedia_sentences_file", type=str, default=None)
    parser.add_argument("--target_entity_pair_file", type=str, default=None)
    parser.add_argument("--pretraining_data_train_file", type=str, default=None)
    parser.add_argument("--pretraining_data_valid_file", type=str, default=None)
    
    parser.add_argument("--n_filter", type=int, default=800000)
    parser.add_argument("--entity_pair_n_thresh", type=int, default=5)
    parser.add_argument("--train_prop", type=float, default=1.0)
    
    args = parser.parse_args()
    return args

def main(args):
    generator = data_builder.PTDGenerator(args)
    generator.run()
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
