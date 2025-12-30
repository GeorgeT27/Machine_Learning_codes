import pandas as pd
import torch
from torch.utils.data import Dataset
def load_data(args,splits):
    df=pd.read_csv(f'{args.data_root}/{splits}.csv')
    texts=df['text'].values.tolist()
    labels=df['label'].values.tolist()
    return texts,labels

class TextDataset(Dataset):
    def __init__(self,data,tokeniser,max_length,is_test):
        self.texts=data[0]
        self.labels=data[1]
        self.tokeniser=tokeniser
        self.max_length=max_length
        self.is_test=is_test
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        text=self.texts[index]
        source=self.tokeniser.batch_encode_plus(
            [text],
            max_length=self.max_length,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) # Tokenizes text (words → subwords);Converts tokens to IDs;Adds special tokens ([CLS], [SEP]);Pads / truncates to a fixed length;Creates attention masks

        source_ids=source['input_ids'].squeeze(0)  # remove batch dimension
        source_mask=source['attention_mask'].squeeze(0)
        data_sample={'source_ids':source_ids,'source_mask':source_mask}
        if not self.is_test:
            label=self.labels[index]
            target_ids=torch.tensor(label,dtype=torch.long)
            data_sample['labels']=target_ids
        return data_sample
    