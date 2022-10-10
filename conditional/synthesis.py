# -*- coding: utf-8 -*-
"""Demonstration script for audio generation using pretrained model.

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

import os

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from torch import nn

from mapping_dict import MappingDict
from models import Mel2Linear, Seq2SeqTransformer
from util import init_mask, make_square_subsequent_mask


class Synthesizer(nn.Module):
    """Synthesizer class for demonstration of audio generation."""

    def __init__(self, cfg):
        """Initialize class."""
        super().__init__()

        self.mapping_dict = MappingDict(cfg)
        self.model = {"transformer": None, "mel2linear": None}
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(
            os.path.join(
                cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.stats_dir, cfg.training.scaler_file
            )
        )

    def _generate_logmel(self, batch):
        """Perform decoding to generate log mel-spectrogram from onomatopoeia.

        Args:
            batch (dict)
                - source (Tensor): onomatopoeia [batch_size, src_len, vocab_size]
                - n_mels (int): number of Mel bands.
                - n_frame (int): number of frame for audio generation.

        Returns:
            refined (Tensor): decoded mel-spectrograms.
        """
        source = batch["source"]
        n_mels = batch["n_mels"]
        n_frame = batch["n_frame"]
        event_label = batch["event_label"]

        device = source.device
        batch_size = source.shape[0]  # assumed to be 1
        masks = init_mask()
        memory = self.model["transformer"].encode(source, event_label, masks)
        target = torch.ones((batch_size, 1, n_mels), device=device)
        for _ in range(n_frame):
            masks["tgt_mask"] = make_square_subsequent_mask(target.shape[1]).to(device)
            decoded = self.model["transformer"].decode(
                target, memory, event_label, masks
            )
            logmel = self.model["transformer"].postnet_proj(decoded)
            target = torch.cat([target, logmel[:, -1:, :]], dim=1)  # append last frame

        logmel = target[:, 1:, :]  # drop <BOS>
        refined = logmel + self.model["transformer"].postnet_residual(logmel)
        return refined

    @torch.no_grad()
    def _generate_audio(self, onomatopoeia, sound_event, n_frame):
        """Perform a inference step to generate audio data.

        Args:
            source (Tensor): onomatopoeia [batch_size, src_len, vocab_size]
            sound_event (Tensor): one-hot vector of sound event label
            n_frame (int): number of frame for audio generation.

        Returns:
            audio (numpy.ndarray): generated audio.
        """
        onomatopoeia = onomatopoeia.to(self.device)
        stats_dir = os.path.join(
            self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.stats_dir
        )
        scaler = joblib.load(os.path.join(stats_dir, self.cfg.training.scaler_file))
        event_label = sound_event.to(self.device).long()
        event_label = F.one_hot(event_label, num_classes=len(self.cfg.sound_event))
        batch = {
            "source": onomatopoeia.to(self.device),
            "event_label": event_label,
            "n_mels": self.cfg.feature.n_mels,
            "n_frame": n_frame,
        }
        logmel = self._generate_logmel(batch)
        loglinear = self.model["mel2linear"](logmel)  # CBHG-based module
        loglinear = loglinear.to("cpu").detach().numpy().copy()  # tensor to numpy
        loglinear = loglinear.squeeze(0)
        loglinear = scaler.inverse_transform(loglinear)
        spec = np.exp(loglinear * np.log(10)).T  # log-scale -> linear-scale
        audio = librosa.griffinlim(  # convert spectrogram into audio
            spec,
            n_fft=self.cfg.feature.n_fft,
            hop_length=self.cfg.feature.hop_length,
            win_length=self.cfg.feature.win_length,
            n_iter=self.cfg.feature.n_iter,
        )
        audio = librosa.util.normalize(audio)

        return audio

    def _get_onoma_tensor(self, onomatopoeia):
        """Get tensor of onomatopoeia.

        Args:
            onomatopoeia (str): onomatopoeia

        Returns:
            onoma_tensor (Torch.tensor): tensor of onomatopoeia
        """
        chars = onomatopoeia.split()
        char_ids = [self.mapping_dict.char2id[c] for c in chars]
        onoma_tensor = torch.from_numpy(np.array(char_ids)).unsqueeze(0)
        return onoma_tensor

    def _get_event_tensor(self, sound_event):
        """Get tensor of sound event label."""
        sound_event = torch.tensor(self.cfg.sound_event.index(sound_event))
        sound_event = sound_event.unsqueeze(0)
        return sound_event

    def forward(self, onomatopoeia, sound_event, n_frame=32):
        """Synthesize audio from onomatopoeia with a trained model.

        Args:
            onomatopoeia (str): onomatopoeia
            sound_event (str): sound event
            n_frame (int): number of frames for audio to be synthesized

        Returns:
            audio (numpy.array): generated audio data
            sample_rate (int): sampling rate in Hz
        """
        self.model["transformer"].eval()
        self.model["mel2linear"].eval()
        onoma_tensor = self._get_onoma_tensor(onomatopoeia)
        event_label = self._get_event_tensor(sound_event)
        audio = self._generate_audio(onoma_tensor, event_label, n_frame)
        sample_rate = self.cfg.feature.sample_rate
        return audio, sample_rate

    def load(self, checkpoint_transformer, checkpoint_mel2linear):
        """Load model parameters."""
        self.mapping_dict.load()
        checkpoint = torch.load(checkpoint_transformer)
        char2id = self.mapping_dict.char2id
        self.model["transformer"] = Seq2SeqTransformer(char2id, self.cfg, self.device)
        self.model["transformer"].load_state_dict(checkpoint)
        checkpoint = torch.load(checkpoint_mel2linear)
        self.model["mel2linear"] = Mel2Linear(self.cfg, self.device)
        self.model["mel2linear"].load_state_dict(checkpoint)


def _get_wavpath(onomatopoeia, root_dir, gen_dir, basename, sound_event):
    """Get path for audio."""
    gen_dir = os.path.join(root_dir, gen_dir)
    os.makedirs(gen_dir, exist_ok=True)
    chars = onomatopoeia.split()
    onoma_seq = "".join(chars)
    wavpath = os.path.join(gen_dir, basename)
    wavpath = wavpath + f"_{onoma_seq}_{sound_event}.wav"
    return wavpath


def _main(cfg):
    """Perform audio synthesis and write audio data to disk."""
    synthesizer = Synthesizer(cfg)
    chechpoint_transformer = cfg.demo.checkpoint.transformer
    chechpoint_mel2linear = cfg.demo.checkpoint.mel2linear
    synthesizer.load(chechpoint_transformer, chechpoint_mel2linear)

    # Perform audio synthesis from onomatopoeia with sound event label
    onomatopoeia = cfg.demo.onomatopoeia  # ex. "b i i i i i"
    sound_event = cfg.demo.sound_event  # ex. "shaver"
    n_frame = cfg.demo.n_frame  # number of frames (= lengths of audio data)
    audio, sample_rate = synthesizer.forward(onomatopoeia, sound_event, n_frame)

    root_dir = cfg.RWCP_SSD.root_dir
    gen_dir = cfg.demo.gen_dir  # directory to save generated audio data
    basename = cfg.demo.basename  # basename for filename of audio data
    wavpath = _get_wavpath(onomatopoeia, root_dir, gen_dir, basename, sound_event)
    sf.write(wavpath, audio, sample_rate)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    _main(config)
