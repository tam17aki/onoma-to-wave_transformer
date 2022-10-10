# -*- coding: utf-8 -*-
"""Python modules for preprocessing of Onoma-to-Wave with Transformer.

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

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from hydra import compose, initialize
from omegaconf import DictConfig
from progressbar import progressbar as prg
from sklearn.preprocessing import StandardScaler

from mapping_dict import MappingDict
from util import AudioAnalysisConfig, logmelspectrogram, logspectrogram


def _make_symlink(src, dest):
    """Make forced symbolic links."""
    try:
        os.symlink(src, dest)
    except FileExistsError:
        os.remove(dest)
        os.symlink(src, dest)


def raw_to_wav(cfg: DictConfig):
    """Convert raw to wav (RIFF header attachment)."""
    rawdir = os.path.join(
        cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.data_dir, "nospeech", "drysrc"
    )

    wavdir = os.path.join(
        cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.wav_dir, "nospeech", "drysrc"
    )

    raw_filename_list = [
        path
        for path in glob.glob(rawdir + "/**", recursive=True)
        if ("16khz" in path and path.endswith(".raw"))
    ]

    for raw_file in prg(raw_filename_list):
        wavdir_plain = os.path.dirname(raw_file.replace(rawdir, wavdir))
        os.makedirs(wavdir_plain, exist_ok=True)
        wave_file = os.path.join(
            wavdir_plain, os.path.basename(raw_file.replace("raw", "wav"))
        )
        data, sample_rate = sf.read(
            raw_file, channels=1, samplerate=cfg.feature.sample_rate, subtype="PCM_16"
        )
        sf.write(wave_file, data, sample_rate)


def make_symlink(cfg: DictConfig):
    """Make symbolic links."""
    wavdir = os.path.join(
        cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.wav_dir, "nospeech", "drysrc"
    )
    wavdir_train = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.wav_dir, "train")
    wavdir_test = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.wav_dir, "test")
    os.makedirs(wavdir_train, mode=0o755, exist_ok=True)
    os.makedirs(wavdir_test, mode=0o755, exist_ok=True)

    var = {"wav_filename_list": [], "symlink_train": "", "symlink_test": ""}

    var["wav_filename_list"] = [
        path
        for path in glob.glob(wavdir + "/**", recursive=True)
        if ("16khz" in path and path.endswith(".wav"))
    ]

    for wav_file in prg(var["wav_filename_list"], prefix="Create symbolic links"):

        # destination (file name)
        var["symlink_train"] = wav_file.replace(wavdir, wavdir_train)
        var["symlink_test"] = wav_file.replace(wavdir, wavdir_test)

        # destination (directory name)
        os.makedirs(os.path.dirname(var["symlink_train"]), mode=0o755, exist_ok=True)
        os.makedirs(os.path.dirname(var["symlink_test"]), mode=0o755, exist_ok=True)

        if any(basename in var["symlink_test"] for basename in cfg.test_basename):
            _make_symlink(src=wav_file, dest=var["symlink_test"])
        else:
            _make_symlink(src=wav_file, dest=var["symlink_train"])


def make_csv(cfg: DictConfig):
    """Make csv files."""
    dir_info = {
        "wav_dir": os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.wav_dir),
        "jp_dir": os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_jpdir),
        "en_dir": os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_endir),
        "train_dir": os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_traindir),
        "test_dir": os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_testdir),
    }
    os.makedirs(dir_info["train_dir"], exist_ok=True)
    os.makedirs(dir_info["test_dir"], exist_ok=True)

    var = {
        "jp_path": glob.glob(dir_info["jp_dir"] + "/**", recursive=True),
        "en_path": glob.glob(dir_info["en_dir"] + "/**", recursive=True),
        "jp_file": [],
        "en_file": [],
        "jp_tmp": [],
        "en_tmp": [],
        "df_jp": None,
        "df_en": None,
    }

    for jp_path, en_path in prg(
        zip(var["jp_path"], var["en_path"]),
        total=len(var["jp_path"]),
        prefix="Pack information into CSV file",
    ):

        var["jp_tmp"] = jp_path.split("/")
        # var["jp_tmp"] = ['', 'work', 'tamamori', 'onomato-wave', 'data',
        # 'RWCPSSD_Onomatopoeia', 'RWCP_SSD_Onomatopoeia_jp', 'nospeech', 'drysrc',
        # 'a1', 'wood3', '052.ono']
        var["jp_file"] = os.path.join(
            # 例 'a1', 'wood3', '052.ono'
            var["jp_tmp"][-3],
            var["jp_tmp"][-2],
            var["jp_tmp"][-1],
        )

        var["en_tmp"] = en_path.split("/")
        var["en_file"] = os.path.join(
            var["en_tmp"][-3], var["en_tmp"][-2], var["en_tmp"][-1]
        )

        if (
            jp_path.endswith(".ono")
            and en_path.endswith(".ono")
            and (var["jp_file"] == var["en_file"])
        ):
            # Onomatopoeia (written in Kana)
            # ex: ポンッ
            var["df_jp"] = pd.read_csv(jp_path, header=None)
            # Leave only the onomatopoeia column.
            df_ono_jp = var["df_jp"].drop(var["df_jp"].columns[[0, 1, 3]], axis=1)

            # Phonetic notation of Onomatopoeia (wrtten in English characters)
            # ex.: p o N q
            var["df_en"] = pd.read_csv(en_path, header=None)
            # Leave only the onomatopoeia column.
            df_ono_en = var["df_en"].drop(var["df_en"].columns[[0, 1, 3]], axis=1)

            # Acoustic event
            # ex.: cherry1
            df_event = pd.DataFrame([var["jp_tmp"][-2] for _ in range(len(df_ono_jp))])

            # Path of the wav file corresponding to the onomatopoeia
            dir_name, file_name = os.path.split(jp_path)
            wav_path = os.path.join(
                dir_name.replace(dir_info["jp_dir"], dir_info["wav_dir"]),
                "16khz",
                file_name.replace(".ono", ".wav"),
            )

            df_path = pd.DataFrame([wav_path for _ in range(len(df_ono_jp))])

            # Concatenate onomatopoeia (Kana), onomatopoeia (English), acoustic events,
            # and wav file paths
            df_concat = pd.concat([df_ono_jp, df_ono_en, df_path, df_event], axis=1)

            if any(x in jp_path for x in cfg.test_basename):
                output_path = jp_path.replace(
                    dir_info["jp_dir"], dir_info["test_dir"]
                ).replace(".ono", ".csv")
            else:
                output_path = jp_path.replace(
                    dir_info["jp_dir"], dir_info["train_dir"]
                ).replace(".ono", ".csv")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_concat.to_csv(output_path, header=False, index=False)


def extract_spec(cfg: DictConfig):
    """Extract spectrograms from audio."""
    spec_list = []
    melspec_list = []
    wav_dir = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.wav_dir, "train")
    wav_files = [
        os.path.abspath(wavfile)
        for wavfile in glob.glob(wav_dir + "/**/*.wav", recursive=True)
        if any(event_name in wavfile for event_name in cfg.sound_event)
    ]
    for wav_file in prg(wav_files, prefix="Extract spectrograms: "):
        audio, sample_rate = librosa.load(
            wav_file, sr=cfg.feature.sample_rate, mono=True
        )
        audio = librosa.util.normalize(audio)
        analysis_config = AudioAnalysisConfig(
            sample_rate=sample_rate,
            n_fft=cfg.feature.n_fft,
            win_length=cfg.feature.win_length,
            hop_length=cfg.feature.hop_length,
            n_mels=cfg.feature.n_mels,
            clip=cfg.feature.min_clip,
        )
        logspec = logspectrogram(audio, analysis_config)
        spec_list.append(logspec)
        logmel = logmelspectrogram(audio, analysis_config)
        melspec_list.append(logmel)

    return spec_list, melspec_list


def compute_standardization(cfg: DictConfig, spec_list, melspec_list, axis=0):
    """Compute mean and standard dev."""
    # Concatenate along time-axis
    spec = np.concatenate(spec_list, axis=axis)
    melspec = np.concatenate(melspec_list, axis=axis)

    scaler = StandardScaler()
    scaler.fit(spec)

    scaler_logmel = StandardScaler()
    scaler_logmel.fit(melspec)

    for i, data in enumerate(spec_list):
        spec_list[i] = scaler.transform(data)

    for i, data in enumerate(melspec_list):
        melspec_list[i] = scaler_logmel.transform(data)

    # Save scaler
    stats_dir = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(stats_dir, cfg.training.scaler_file))

    return spec_list, melspec_list


def save_specs(cfg: DictConfig, spec_list, melspec_list):
    """Save extracted spectrograms."""
    feats_dir = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.feats_dir)
    os.makedirs(feats_dir, exist_ok=True)
    joblib.dump(spec_list, os.path.join(feats_dir, cfg.preprocess.spec_file))
    joblib.dump(melspec_list, os.path.join(feats_dir, cfg.preprocess.melspec_file))


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    raw_to_wav(config)
    make_csv(config)
    make_symlink(config)

    specs, melspecs = extract_spec(config)
    specs, melspecs = compute_standardization(config, specs, melspecs)
    save_specs(config, specs, melspecs)

    mapping_dict = MappingDict(config)
    mapping_dict.make()
    mapping_dict.save()
