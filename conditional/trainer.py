# -*- coding: utf-8 -*-
"""Trainer class definition for Onoma-to-Wave model (conditioned on sound events).

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
from progressbar import progressbar
from torch import optim

from models import get_model
from scheduler import TransformerLR
from util import (
    create_mask,
    get_nframes,
    init_mask,
    make_nonpad_mask,
    make_square_subsequent_mask,
    put_bos,
)


class Trainer:
    """Trainer Class."""

    def __init__(self, cfg, mapping_dict):
        """Initialize class."""
        self.mapping_dict = mapping_dict
        self.cfg = cfg
        self.model = {"transformer": None, "mel2linear": None}
        self.optimizer = {"transformer": None, "mel2linear": None}
        self.scheduler = {"transformer": None, "mel2linear": None}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _configure_optimizers(self, phase):
        """Instantiate optimizer.

        Args:
            phase (str): specify training phase; 'transformer' or 'mel2linear'.
        """
        if phase == "transformer":
            optimizer = optim.RAdam(
                self.model[phase].parameters(),
                lr=self.cfg.training.learning_rate[phase],
                betas=(0.9, 0.98),  # magic numbers from the original transformer paper
                eps=1e-09,  # a magic number from the original transformer paper
            )
        else:
            optimizer = optim.RAdam(
                self.model[phase].parameters(),
                lr=self.cfg.training.learning_rate[phase],
                betas=(0.9, 0.98),  # magic numbers from the original transformer paper
                eps=1e-09,  # a magic number from the original transformer paper
            )
        self.optimizer[phase] = optimizer
        if self.cfg.training.use_scheduler:
            if self.cfg.training.fit.transformer is True:
                scheduler = TransformerLR(optimizer, self.cfg.training.warmup_epochs)
            else:
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    gamma=self.cfg.training.gamma,
                    milestones=self.cfg.training.milestones,
                )
            self.scheduler[phase] = scheduler

    def _training_step(self, batch, phase):
        """Perform a training step to compute loss.

        B = batch_size, S = src_len, V = vocab_size, T = tgt_len, M = n_mels
        N = n_fft // 2 + 1

        Args:
            batch (dict)
            if phase == 'transformer':
                - source (Tensor) : onomatopoeia [B, S, V]
                - src_lengths (list): list of sequence length of onomatopoeia
                - target (Tensor): log mel-spectrograms [B, T, M]
                - tgt_lengths (list): list of frame lengths of mel-spectrograms
                - event_label (Tensor) : one-hot vector of sound event label
            if phase == 'mel2linear':
                - source (Tensor): log mel-spectrogram [B, S, M]
                - target (Tensor): log linear spectrogram [B, T, N]
                - tgt_lengths (list): list of frame lengths in mini-batch

        Returns:
            loss: training loss.
        """
        if phase == "transformer":
            source = batch["source"]  # onomatopoeia
            src_lengths = batch["src_lengths"]
            target = batch["target"]
            tgt_lengths = batch["tgt_lengths"]
            event_label = batch["event_label"]

            # after dropping the last frame, put <BOS> at the first frame
            tgt_input = put_bos(target[:, :-1, :])

            # make masks for computing multi-head self-attention
            masks = create_mask(tgt_input, src_lengths, tgt_lengths)

            # convert onomatopoeia into log mel-spectrogram via Transformer
            logmel, refined = self.model[phase].forward(
                source, tgt_input, event_label, masks
            )

            # make a mask to exclude padding frames in computing loss
            mask = make_nonpad_mask(tgt_lengths).unsqueeze(-1).to(source.device)
            logmel = logmel.masked_select(mask)
            refined = refined.masked_select(mask)
            target = target.masked_select(mask)
            loss = F.l1_loss(logmel, target) + F.l1_loss(refined, target)
        else:
            logmel = batch["source"]
            target = batch["target"]
            tgt_lengths = batch["tgt_lengths"]

            # convert log mel-spectrogram into log linear spectrogram
            loglinear = self.model[phase].forward(logmel)

            # make a mask to exclude padding frames in computing loss
            mask = make_nonpad_mask(tgt_lengths).unsqueeze(-1).to(target.device)
            loglinear = loglinear.masked_select(mask)
            target = target.masked_select(mask)
            loss = F.l1_loss(loglinear, target)

        return loss

    def _training_epoch(self, dataloader, phase):
        """Perform a training epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader for training.
            phase (str): specify training phase; 'transformer' or 'mel2linear'.
        """
        epoch_loss = 0
        for (
            onomatopoeia,
            seq_lengths,
            specs,
            melspecs,
            frame_lengths,
            event_label,
        ) in dataloader:
            self.optimizer[phase].zero_grad()
            if phase == "transformer":
                event_label = event_label.to(self.device).long()
                event_label = F.one_hot(
                    event_label, num_classes=len(self.cfg.sound_event)
                )
                batch = {
                    "source": onomatopoeia.to(self.device),
                    "src_lengths": seq_lengths,
                    "target": melspecs.to(self.device).float(),
                    "tgt_lengths": frame_lengths,
                    "event_label": event_label,
                }
            else:
                batch = {
                    "source": melspecs.to(self.device).float(),
                    "target": specs.to(self.device).float(),
                    "tgt_lengths": frame_lengths,
                }

            loss = self._training_step(batch, phase)
            epoch_loss += loss.item()
            loss.backward()
            if self.cfg.training.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model[phase].parameters(), self.cfg.training.grad_max_norm
                )
            self.optimizer[phase].step()

        if self.cfg.training.use_scheduler:
            self.scheduler[phase].step()

        epoch_loss = epoch_loss / len(dataloader)
        return epoch_loss

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
    def _generate_audio(self, batch):
        """Perform a inference step to generate audio data."""
        onomatopoeia, wavfiles, event_label = batch
        stats_dir = os.path.join(
            self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.stats_dir
        )
        scaler = joblib.load(os.path.join(stats_dir, self.cfg.training.scaler_file))

        # Generate standardized log-scale mel-spectrogram from onomatopoeia
        event_label = event_label.to(self.device).long()
        event_label = F.one_hot(event_label, num_classes=len(self.cfg.sound_event))
        mini_batch = {
            "source": onomatopoeia.to(self.device),
            "n_mels": self.cfg.feature.n_mels,
            "n_frame": get_nframes(
                wavfiles[0],
                self.cfg.feature.sample_rate,
                self.cfg.feature.win_length,
                self.cfg.feature.hop_length,
            ),
            "event_label": event_label,
        }
        logmel = self._generate_logmel(mini_batch)
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

    def _write_audio(self, batch, audio):
        """Write generated audio data to disk."""
        onomatopoeia, wavfiles, _ = batch

        wav_dir = os.path.join(self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.wav_dir)
        gen_dir = os.path.join(self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.gen_dir)
        os.makedirs(gen_dir, exist_ok=True)

        audio_dir = wavfiles[0].replace(wav_dir, gen_dir)
        dirname = os.path.dirname(audio_dir)
        os.makedirs(dirname, exist_ok=True)

        basename = os.path.splitext(os.path.basename(audio_dir))[0]
        event_name = wavfiles[0].split("/")[-3]

        # phoneme sequence in numerical value
        onoma_idlist = onomatopoeia[0].detach().numpy().copy()

        # phoneme sequence in character
        translation = "".join(
            list(map(lambda id: self.mapping_dict.id2char[id], onoma_idlist))
        )

        # fix pathname for audio file
        basename = f"{event_name}_" + basename + f"_{translation}.wav"
        audio_path = os.path.join(dirname, basename)

        sf.write(audio_path, audio, self.cfg.feature.sample_rate)

    def _save(self, phase):
        """Save model parameters.

        Args:
            phase (str): specify training phase; 'transformer' or 'mel2linear'.
        """
        model_dir = os.path.join(
            self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.model_dir
        )
        os.makedirs(model_dir, exist_ok=True)
        n_epoch = self.cfg.training.n_epoch[phase]
        n_batch = self.cfg.training.n_batch
        learning_rate = self.cfg.training.learning_rate[phase]
        warmup = self.cfg.training.warmup_epochs
        prefix = self.cfg.training.model_prefix
        if phase == "transformer":
            model_file = os.path.join(
                model_dir,
                f"{prefix}_epoch{n_epoch}_batch{n_batch}"
                f"_lr{learning_rate}_warmup{warmup}_{phase}.pt",
            )
        else:
            model_file = os.path.join(
                model_dir,
                f"{prefix}_epoch{n_epoch}_batch{n_batch}_lr{learning_rate}_{phase}.pt",
            )
        torch.save(self.model[phase].state_dict(), model_file)

    def _load(self, phase):
        """Load model parameters.

        Args:
            phase (str): specify training phase; 'transformer' or 'mel2linear'.
        """
        model_dir = os.path.join(
            self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.model_dir
        )
        n_epoch = self.cfg.training.n_epoch[phase]
        n_batch = self.cfg.training.n_batch
        learning_rate = self.cfg.training.learning_rate[phase]
        warmup = self.cfg.training.warmup_epochs
        prefix = self.cfg.training.model_prefix
        if phase == "transformer":
            model_file = os.path.join(
                model_dir,
                f"{prefix}_epoch{n_epoch}_batch{n_batch}"
                f"_lr{learning_rate}_warmup{warmup}_{phase}.pt",
            )
        else:
            model_file = os.path.join(
                model_dir,
                f"{prefix}_epoch{n_epoch}_batch{n_batch}_lr{learning_rate}_{phase}.pt",
            )
        checkpoint = torch.load(model_file)
        self.model[phase].load_state_dict(checkpoint)

    def fit(self, phase, dataloader):
        """Perform model training.

        phase (str): specify training phase; 'transformer' or 'mel2linear'.
        dataloader (torch.utils.data.DataLoader): dataloader for training.
        """
        torch.backends.cudnn.benchmark = True  # a magic
        self.model[phase] = get_model(self.cfg, phase, self.mapping_dict, self.device)
        self._configure_optimizers(phase)
        self.model[phase].train()  # turn on train mode
        for epoch in progressbar(
            range(1, self.cfg.training.n_epoch[phase] + 1),
            prefix=f"Model training ({phase}): ",
            suffix="\n",
        ):
            epoch_loss = self._training_epoch(dataloader, phase)
            print(f"Epoch {epoch}: loss = {epoch_loss:.6f}")

        self._save(phase)  # save model parameters

    def generate(self, dataloader):
        """Audio generation with trained models.

        dataloader (torch.utils.data.DataLoader): dataloader for inference.
        """
        for phase in self.model:
            self.model[phase] = get_model(
                self.cfg, phase, self.mapping_dict, self.device
            )
            self._load(phase)  # load model parameters
            self.model[phase].eval()  # turn on eval mode

        for batch in progressbar(
            dataloader, prefix="Audio generation with trained models "
        ):
            audio = self._generate_audio(batch)
            self._write_audio(batch, audio)
