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
