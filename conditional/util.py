# -*- coding: utf-8 -*-
"""Utility functions for Onoma-to-Wave model (conditioned on sound events).

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
from typing import NamedTuple, Optional

import librosa
import numpy as np
import torch
from torch import nn


class AudioAnalysisConfig(NamedTuple):
    """Class for audio analysis configuration.

    Args:
        sample_rate_ (int):sampling rate in Hz..
        n_fft (int, optional): FFT size.
        hop_length (int, optional): hop length. defaults to 12.5ms.
        win_length (int, optional): Window length. defaults to 50 ms.
        n_mels (int, optional): Number of mel bins. defaults to 80.
        fmin (int, optional): minimum frequency. defaults to 0.
        fmax (int, optional): maximum frequency. defaults to sr / 2.
        clip (float, optional): clip the magnitude. defaults to 0.0001.
    """

    sample_rate: int = 16000
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    fmin: Optional[int] = None
    fmax: Optional[int] = None
    clip: float = 0.0001


def _next_power_of_2(size):
    return 1 if size == 0 else 2 ** (size - 1).bit_length()


# adapted from
# https://github.com/r9y9/ttslearn/blob/master/ttslearn/dsp.py
def logspectrogram(audio, config):
    """Compute log-spectrogram.

    Args:
        audio (ndarray): Waveform.
        config (AudioAnalysisConfig):
            sample_rate (int): Sampling rate.
            n_fft (int, optional): FFT size.
            hop_length (int, optional): Hop length. Defaults to 12.5ms.
            win_length (int, optional): Window length. Defaults to 50 ms.
            clip (float, optional): Clip the magnitude. Defaults to 0.001.

    Returns:
        numpy.ndarray: Log-spectrogram.

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """
    sample_rate = config.sample_rate
    n_fft = config.n_fft
    hop_length = config.hop_length
    win_length = config.win_length
    clip = config.clip

    if hop_length is None:
        hop_length = int(sample_rate * 0.0125)
    if win_length is None:
        win_length = int(sample_rate * 0.050)
    if n_fft is None:
        n_fft = _next_power_of_2(win_length)

    spec = librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    spec = np.maximum(np.abs(spec), clip)
    spec = np.log10(spec)
    return spec.T


# adapted from
# https://github.com/r9y9/ttslearn/blob/master/ttslearn/dsp.py
def logmelspectrogram(audio, config):
    """Compute log-melspectrogram.

    Args:
        audio (ndarray): Waveform.
        config (AudioAnalysisConfig):
            sample_rate (int): Sampling rate.
            n_fft (int, optional): FFT size.
            hop_length (int, optional): Hop length. Defaults to 12.5ms.
            win_length (int, optional): Window length. Defaults to 50 ms.
            n_mels (int, optional): Number of mel bins. Defaults to 80.
            fmin (int, optional): Minimum frequency. Defaults to 0.
            fmax (int, optional): Maximum frequency. Defaults to sr / 2.
            clip (float, optional): Clip the magnitude. Defaults to 0.001.

    Returns:
        numpy.ndarray: Log-melspectrogram.

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """
    sample_rate = config.sample_rate
    n_fft = config.n_fft
    hop_length = config.hop_length
    win_length = config.win_length
    n_mels = config.n_mels
    fmin = config.fmin
    fmax = config.fmax
    clip = config.clip
    if hop_length is None:
        hop_length = int(sample_rate * 0.0125)
    if win_length is None:
        win_length = int(sample_rate * 0.050)
    if n_fft is None:
        n_fft = _next_power_of_2(win_length)

    spec = librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    fmin = 0 if fmin is None else fmin
    fmax = sample_rate // 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels
    )
    spec = np.dot(mel_basis, np.abs(spec))
    spec = np.maximum(spec, clip)
    spec = np.log10(spec)
    return spec.T


def get_nframes(wavfile, sample_rate, frame_length, hop_length):
    """Get the number of frames from wavfile."""
    audio, _ = librosa.load(wavfile, sr=sample_rate, mono=True)
    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return frames.shape[1]


# adapted from
# https://github.com/r9y9/ttslearn/blob/master/ttslearn/util.py
def make_pad_mask(lengths, maxlen=None):
    """Make mask of padding frames.

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        mask (torch.BoolTensor): mask of padding frames.

    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(int(len(lengths)), maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


# adapted from
# https://github.com/r9y9/ttslearn/blob/master/ttslearn/util.py
def make_nonpad_mask(lengths, maxlen=None):
    """Make mask of non-padding frames.

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        mask (BoolTensor): maks of non-padding frames [batch_size, max_len, 1]
        
    Copyright (C) 2021 by Ryuichi Yamamoto
    MIT License.
    """
    return ~make_pad_mask(lengths, maxlen)


def make_square_subsequent_mask(size):
    """Make a subsequent mask for decoder input of transformer.

    Args:
        size (int): size of mask.

    Returns:
        mask (BoolTensor): a subsequent square mask [size, size]
    """
    mask = nn.Transformer.generate_square_subsequent_mask(size).type(torch.bool)
    return mask


def init_mask():
    """Initialize masks for transformer.

    Returns:
        masks (dict): dictionary of masks; all the values are set to None.
    """
    masks = {
        "src_mask": None,  # no mask is needed for source self-attention.
        "tgt_mask": None,  # a subsequent mask is needed for target self-attention.
        "memory_mask": None,  # no mask is needed for source-target attention.
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
    }
    return masks


def create_mask(tgt, src_lengths, tgt_lengths):
    """Create masks for transformer.

    Args:
        tgt (Tensor): mel-spectrogram. [batch_size_size, tgt_len, n_mels]
        src_lengths (list): list of sequence length of onomatopoeia in mini-batch.
        tgt_lengths (list): list of frame length of mel-spectrogram in mini-batch.

    Returns:
        masks (dict): dictionary of mask tensors.
    """
    masks = init_mask()
    device = tgt.device
    masks["tgt_mask"] = make_square_subsequent_mask(tgt.shape[1]).to(device)
    masks["src_key_padding_mask"] = make_pad_mask(src_lengths).to(device)
    masks["tgt_key_padding_mask"] = make_pad_mask(tgt_lengths).to(device)
    masks["memory_key_padding_mask"] = make_pad_mask(src_lengths).to(device)

    return masks


def put_bos(melspecs):
    """Put the <BOS> tensor at the first frame."""
    bos = melspecs.new_zeros((melspecs.shape[0], 1, melspecs.shape[2]))
    melspecs = torch.cat([bos, melspecs], dim=1)
    return melspecs
