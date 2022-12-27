import argparse

from module.techqa_data_generator import data_builder

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--building_predefined_entities_list", action="store_true")
    parser.add_argument("--preprocess_techqa", action="store_true")
    parser.add_argument("--split_train_valid", action="store_true")
    parser.add_argument("--get_stats", action="store_true")
    
    parser.add_argument("--use_title", action="store_true")
    
    parser.add_argument("--techqa_corpus_file", type=str, default=None)
    parser.add_argument("--entity_file", type=str, default=None)
    parser.add_argument("--eid2n_file", type=str, default=None)
    parser.add_argument("--target_entity_pair_file", type=str, default=None)
    parser.add_argument("--techqa_corpus_preprocessed_file", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--valid_file", type=str, default=None)
    
    parser.add_argument("--n_entity_thresh", type=int, default=5)
    parser.add_argument("--train_prop", type=float, default=0.9)
    
    args = parser.parse_args()
    return args

def main(args):
    generator = data_builder.TechQAPreprocessor(args)
    generator.run()
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
