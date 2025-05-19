'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

### TORCH TEXT IS LEGACY: pip install torchtext==0.10.0
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import os, re
import pickle

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    MAX_SEQ = 200
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        # This puts us in the text_classification folder, structured as follows:
        # . -> README, test, train
        # train/test -> {idx}_{rating}.txt

        parent_path = self.dataset_source_folder_path if self.dataset_source_folder_path else "data/stage_4_code/text_classification"
        data = {
            "train": {"X": [], "y": []},
            "test": {"X": [], "y": []}
        }

        tokenizer = get_tokenizer('basic_english')
        all_tokens = []

        for split in ["train", "test"]:
            for rating in ["pos", "neg"]:
                folderpath = f"{parent_path}/{split}/{rating}"
                for reviewfile in os.listdir(folderpath):
                    with open(f"{folderpath}/{reviewfile}") as f:
                        raw = f.read()
                        processed_text = self.preprocess(raw)
                        tokens = tokenizer(processed_text)
                        all_tokens.append(tokens)

                        data[split]["X"].append(tokens)
                        data[split]["y"].append(1 if rating=="pos" else 0)

        vocab = build_vocab_from_iterator(all_tokens, specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        data["train"]["X"] = [self.to_index(x, vocab) for x in data["train"]["X"]]
        data["test"]["X"] = [self.to_index(x, vocab) for x in data["test"]["X"]]           
        return data, vocab

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", '', text)
        text = re.sub(r"\s+", ' ', text)
        text = re.sub(r"\d", '', text)
        return text

    def to_index(self, tokens, vocab):
        idxs = vocab(tokens)
        return idxs[:self.MAX_SEQ] + [vocab["<pad>"]] * max(0, self.MAX_SEQ - len(idxs)) 
        


