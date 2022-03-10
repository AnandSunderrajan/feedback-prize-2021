import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn import functional as F
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import bitsandbytes as bnb

from utils import EarlyStopping, prepare_training_data, target_id_map

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=str, default="../model", required=False)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, required=False)
    parser.add_argument("--optimizer", type=int, default=32, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    return parser.parse_args()
