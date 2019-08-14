import pandas as pd
from .regression_processor import RegressionProcessor
import numpy as np
import torch
from torch.utils.data import (DataLoader, TensorDataset)
import os
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from multiprocessing import Pool, cpu_count
from .convert_examples_to_features import convert_example_to_feature
from tqdm import tqdm



def preprocess(text, multiprocessing, Process_count, chunksize):
    text1 = pd.DataFrame(text, columns=['body'])
    train_df_bert = pd.DataFrame({
        'id': range(len(text)),
        'label': 0,
        'alpha': ['a'] * text1.shape[0],
        'text': text1['body'].replace(r'\n', ' ', regex=True)
    })
    train_df_bert.to_csv('train_regression.tsv', sep='\t', index=False, header=False)

    processor = RegressionProcessor()
    train_examples = processor.get_train_examples('')
    train_examples_len = len(train_examples)
    os.remove('train_regression.tsv')

    OUTPUT_MODE = 'regression'
    MAX_SEQ_LENGTH = 128

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    label_map = {'0.0': 0}
    train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in
                                     train_examples]

    process_count = Process_count
    if multiprocessing == 'False':
        with Pool(process_count) as p:
            train_features = list(tqdm(
                p.imap(convert_example_to_feature, train_examples_for_processing),
                total=train_examples_len))
    elif multiprocessing == 'True':
        with Pool(process_count) as p:
            train_features = list(tqdm(
                p.imap(convert_example_to_feature, train_examples_for_processing, chunksize),
                total=train_examples_len))
    return train_features