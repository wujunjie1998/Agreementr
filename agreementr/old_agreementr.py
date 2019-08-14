import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, TensorDataset)
import os
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from multiprocessing import Pool, cpu_count
import convert_examples_to_features
from tqdm import tqdm
from regression_processor import RegressionProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(text):

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

    process_count = cpu_count() - 1
    if __name__ == '__main__':
        if train_examples_len > 10000:
            with Pool(process_count) as p:
                train_features = list(tqdm(
                    p.imap(convert_examples_to_features.convert_example_to_feature, train_examples_for_processing),
                    total=train_examples_len))
        else:
            with Pool(process_count) as p:
                train_features = list(tqdm(
                    p.imap(convert_examples_to_features.convert_example_to_feature, train_examples_for_processing),
                    total=train_examples_len))

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

    # using 'cpu' to predict
    if device.type == 'cpu':
        model_state_dict = torch.load(
            'data/agreement_model.bin', map_location='cpu')
    # else: using 'gpu'
    else:
        model_state_dict = torch.load(
            'data/agreement_model.bin')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', state_dict=model_state_dict,
                                                          num_labels=1)
    # model = DataParallel(model)
    model.to(device)
    model.eval()

    preds = []
    bert_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    bert_dataloader = DataLoader(bert_data, batch_size=256)

    if device.type == 'cpu':
        for step, batch in enumerate(tqdm(bert_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            # logits = logits.detach().cpu()
            if len(preds) == 0:
                preds.append(logits)
            else:
                preds[0] = np.append(
                    preds[0], logits, axis=0)
        preds = [item for sublist in preds for item in sublist]
        preds = np.squeeze(np.array(preds))
    else:
        for step, batch in enumerate(tqdm(bert_dataloader)):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            logits = logits[0]
            logits = logits.detach().cpu()
            if len(preds) == 0:
                preds.append(logits)
            else:
                preds[0] = np.append(
                    preds[0], logits, axis=0)
        preds = [item for sublist in preds for item in sublist]
        preds = np.squeeze(np.array(preds))
    return preds


preds = predict(['how much keys for victor black Tachyon ?'])
print(preds)