# -*- coding: utf-8 -*-
"""Mapping dictionary class for Onoma-to-Wave model.

Copyright (C) 2022 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import glob
import os
import pickle

import pandas as pd
from omegaconf import DictConfig
from progressbar import progressbar as prg


class MappingDict:
    """Dictionaries that maps phoneme symbols to serial numbers (ids) and vice versa."""

    def __init__(self, cfg: DictConfig):
        """Initialize class."""
        self.dir_info = {
            "train_dir": os.path.join(
                cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_traindir
            ),
            "test_dir": os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_testdir),
            "dict_dir": os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.dict_dir),
        }

        self.cfg = cfg
        self._char2id = {}  # a mapping dictionary from phonemes to serial numbers (ids)
        self._id2char = {}  # a mapping dictionary from serial numbers (ids) to phonemes

    def make(self):
        """Create dictionaries."""
        train_csv = [
            csv_file
            for csv_file in glob.glob(
                self.dir_info["train_dir"] + "/**/*.csv", recursive=True
            )
            if any(event_name in csv_file for event_name in self.cfg.sound_event)
        ]

        test_csv = [
            csv_file
            for csv_file in glob.glob(
                self.dir_info["test_dir"] + "/**/*.csv", recursive=True
            )
            if any(event_name in csv_file for event_name in self.cfg.sound_event)
        ]

        self._char2id.update({"<BOS>": 0, " ": 1})

        for word in prg(train_csv, prefix="Dictionary creation from training data "):
            dataframe = pd.read_csv(word, header=None)
            for i in range(len(dataframe)):

                # phoneme sequence (delimited space; ex. 'p o N q')
                ono_en = dataframe.iat[i, 1]
                for phone in ono_en.split():  # phonemic symbols
                    if phone not in self._char2id:  # first appearance
                        self._char2id[phone] = len(self._char2id)  # add new phoneme

        for word in prg(test_csv, prefix="Dictionary creation from test data "):
            dataframe = pd.read_csv(word, header=None)
            for i in range(len(dataframe)):

                # phoneme sequence (delimited space; ex. p i N)
                ono_en = dataframe.iat[i, 1]
                for phone in ono_en.split():  # phonemic symbols
                    if phone not in self._char2id:  # first appearance
                        self._char2id[phone] = len(self._char2id)  # add new phoneme

        # Create a reverse dictionary
        # Mapping from serial numbers to phonemes
        for key, value in self._char2id.items():
            self._id2char[value] = key

    def save(self):
        """Save the created conversion dictionary."""
        dict_path = os.path.join(
            self.dir_info["dict_dir"], self.cfg.training.mapping_dict
        )
        os.makedirs(self.dir_info["dict_dir"], mode=0o755, exist_ok=True)
        with open(dict_path, "wb") as file:
            pickle.dump({"char2id": self._char2id, "id2char": self._id2char}, file)

    def load(self):
        """Load the created conversion dictionary."""
        dict_path = os.path.join(
            self.dir_info["dict_dir"], self.cfg.training.mapping_dict
        )
        with open(dict_path, "rb") as file:
            dictionary = pickle.load(file)

        self._char2id = dictionary["char2id"]
        self._id2char = dictionary["id2char"]

    @property
    def char2id(self):
        """Get char2id member."""
        return self._char2id

    @property
    def id2char(self):
        """Get id2char member."""
        return self._id2char
