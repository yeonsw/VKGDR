import argparse

from module.techqa_pair_target_data_generator import data_builder

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preprocess_qa_pairs", action="store_true")
    parser.add_argument("--get_subset", action="store_true")
    
    parser.add_argument("--techqa_qa_file", type=str, default=None)
    parser.add_argument("--entity_file", type=str, default=None)
    parser.add_argument("--q_doc_file", type=str, default=None)
    parser.add_argument("--proportion", type=float, default=None)
    parser.add_argument("--q_doc_subset_file", type=str, default=None)
    
    args = parser.parse_args()
    return args

def main(args):
    generator = data_builder.DataBuilder(args)
    generator.run()
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
