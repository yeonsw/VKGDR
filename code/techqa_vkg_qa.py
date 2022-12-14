import argparse
from dataclasses import dataclass, field
import random
import numpy as np
from transformers import HfArgumentParser
import torch
from typing import Optional

from module.relation_encoder.model import PredEncoderArguments, TechQADocRetriever
from module.relation_encoder.techqa_pred_data import PredDataArguments, TechQAPredDataset

@dataclass
class Additional:
    seed: Optional[int] = field(default=42)

def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    return 0

def main():
    parser = HfArgumentParser((PredDataArguments, PredEncoderArguments, Additional))
    data_args, model_args, additional_args = parser.parse_args_into_dataclasses()
    
    _ = set_seed(additional_args.seed)
    
    model = TechQADocRetriever(model_args, data_args)
    model.pred_model()
    return 0

if __name__ == "__main__":
    _ = main()

