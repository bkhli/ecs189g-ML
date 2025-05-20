'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

### TORCH TEXT IS LEGACY: pip install torchtext==0.16.0
# from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Changed to NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os, re
import pickle
# import contractions

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

        all_tokens = []
        stop_words = set(stopwords.words('english'))

        for split in ["train", "test"]:
            for rating in ["pos", "neg"]:
                folderpath = f"{parent_path}/{split}/{rating}"
                for reviewfile in os.listdir(folderpath):
                    with open(f"{folderpath}/{reviewfile}") as f:
                        raw = f.read()
                        # print(f"\n\n{rating} - {raw}")
                        processed_text = self.preprocess(raw)
                        # print(f"\n\n{rating} - {processed_text}")
                        tokens = word_tokenize(processed_text)
                        words = [word for word in tokens if word.isalpha()]
                        words = [w for w in words if not w in stop_words]
                        # print(f"\n\n{rating} - {tokens}")
                        # print(f"\n\n{rating} - {words}")
                        all_tokens.append(words)

                        data[split]["X"].append(words)
                        data[split]["y"].append(1 if rating=="pos" else 0)

        vocab = build_vocab_from_iterator(all_tokens, specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        data["train"]["X"] = [self.to_index(x, vocab) for x in data["train"]["X"]]
        data["test"]["X"] = [self.to_index(x, vocab) for x in data["test"]["X"]]           
        # print("[DEBUG] Final vocab object before return:", vocab)
        # print("[DEBUG] Type of vocab:", type(vocab))
        return data, vocab

    # Now, this is only used to lowercase and strip html
    def preprocess(self, text):
        text = text.lower()
        # text = contractions.fix(text)
        text = re.sub(r"</?[^>]+>", " ", text) # deletes html tags
        # text = re.sub(r"[^\w\s]", '', text) # Removes all punctuation and special characters
        # text = re.sub(r"\s+", ' ', text) # Collapses multiple whitespace
        # text = re.sub(r"\d", '', text) # Deletes digits. Include or delete?
        return text

    def to_index(self, tokens, vocab):
        idxs = vocab(tokens)
        return idxs[:self.MAX_SEQ] + [vocab["<pad>"]] * max(0, self.MAX_SEQ - len(idxs)) 
        


