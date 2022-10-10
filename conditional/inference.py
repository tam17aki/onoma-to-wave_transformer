# -*- coding: utf-8 -*-
"""Inference script for Onoma-to-Wave model (conditioned on sound events).

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
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from dataset import get_dataloader
from mapping_dict import MappingDict
from trainer import Trainer


def main(cfg: DictConfig):
    """Perform inference."""
    # Dump configuration
    print(OmegaConf.to_yaml(cfg), flush=True)

    # A dictionary that maps phoneme symbols and serial numbers (id) to each other
    mapping_dict = MappingDict(cfg)
    mapping_dict.load()
    dataloader = get_dataloader(cfg, mapping_dict, training=False)

    trainer = Trainer(cfg, mapping_dict)
    trainer.generate(dataloader)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
