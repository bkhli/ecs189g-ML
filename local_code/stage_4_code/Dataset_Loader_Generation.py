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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

### TORCH TEXT IS LEGACY: pip install torchtext==0.16.0
# from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from local_code.base_class.dataset import dataset

# import contractions


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    preview = 3

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
        stop_words = set(stopwords.words("english"))

        reviewfile = "data"
        with open(f"{parent_path}/{reviewfile}") as f:
            for line in f:
                # Example line: 1621,"Why was the tomato blushing? Because it saw the salad dressing!"
                joke_string = line.split(',"')[1][:-1]  # Crops off the last parentheses
                joke_string = joke_string.lower()
                words = word_tokenize(joke_string)
                # words = [word for word in tokens if word.isalpha()]
                # words = [w for w in words if not w in stop_words]
                # print(f"Joke: {words}")

                if len(words) < self.preview + 1:
                    continue

                all_tokens.append(words)

                for i in range(len(words) - self.preview):
                    data["X"].append(words[i : i + self.preview])
                    data["y"].append(words[i + self.preview])
                data["X"].append(words[-self.preview :])
                data["y"].append("<eos>")  # eos -> End of sequence

        for i in range(50):
            print(f'Hint: {data["X"][i]} | Answer {data["y"][i]}')

        vocab = build_vocab_from_iterator(
            all_tokens, specials=["<pad>", "<unk>", "<eos>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        data["X"] = [[vocab[token] for token in seq] for seq in data["X"]]
        data["y"] = [vocab[token] for token in data["y"]]
        return data, vocab
