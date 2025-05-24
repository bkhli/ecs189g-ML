"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pickle
import re

# Changed to NLTK
import nltk

# from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize

### TORCH TEXT IS LEGACY: pip install torchtext==0.16.0
# from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from local_code.base_class.dataset import dataset

# import contractions


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    MAX_SEQ = 100

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        # This puts us in the text_classification folder, structured as follows:
        # . -> README, test, train
        # train/test -> {idx}_{rating}.txt

        parent_path = (
            self.dataset_source_folder_path
            if self.dataset_source_folder_path
            else "data/stage_4_code/text_generation"
        )
        data = {"X": [], "y": []}

        all_tokens = []
        # stop_words = set(stopwords.words("english"))

        reviewfile = "data"
        tokenizer = (
            TweetTokenizer()
        )  # This one works better. the other one splits "Don't" into "Do" and "n't". Confusing

        with open(f"{parent_path}/{reviewfile}") as f:
            for line in f:
                # Example line: { 1621,"Why was the tomato blushing? Because it saw the salad dressing!" }
                # print(line)
                joke_string = (
                    line.split(',"')[1].strip()[:-1].lower()
                )  # Strips \n and an ending "
                # print(joke_string)
                while '""' in joke_string:
                    joke_string = joke_string.replace('""', '"')
                # print(joke_string)

                # words = word_tokenize(joke_string)
                words = tokenizer.tokenize(joke_string)
                if len(words) < 2:
                    continue

                all_tokens.append(words)

                stopper = min(self.MAX_SEQ, len(words))
                for i in range(1, stopper):  # start from 1 instead of fixed preview
                    data["X"].append(words[:i])  # hint of length 1 to len-1
                    data["y"].append(words[i])  # the i-th word is the target

                data["X"].append(words)  # final hint
                data["y"].append("<eos>")

        for i in range(500):
            print(f'Hint: {data["X"][i]} | Answer {data["y"][i]}')

        print(f"Number of rows: {len(data['X'])}")

        vocab = build_vocab_from_iterator(
            all_tokens, specials=["<pad>", "<unk>", "<eos>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        data["X"] = [[vocab[token] for token in seq] for seq in data["X"]]
        data["y"] = [vocab[token] for token in data["y"]]
        return data, vocab, False
