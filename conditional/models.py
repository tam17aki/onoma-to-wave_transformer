# -*- coding: utf-8 -*-
"""Model definitions of Onoma-to-Wave model (conditioned on sound events).

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
from torch import nn

from module import (
    CBHG,
    CBHGConfig,
    DecoderPrenet,
    DecoderPrenetConfig,
    EncoderPrenet,
    EncoderPrenetConfig,
    EventEmbedding,
    Postnet,
    PostnetConfig,
    ScaledPositionalEncoding,
)


class Seq2SeqTransformer(nn.Module):
    """Transformer Class.

    Args:
        char2id (dict): dictionary between phonemes and index.
        config (DictConfig): configuration of model.
        device (torch.device): "cuda" or "cpu".
    """

    def __init__(self, char2id, config, device):
        """Initialize class."""
        super().__init__()
        enc_prenet_config = EncoderPrenetConfig(
            char2id[" "],
            len(char2id),
            config.model.enc_prenet.emb_size,
            config.model.transformer.attention_dim,
            config.model.enc_prenet.conv_channels,
            config.model.enc_prenet.kernel_size,
            config.model.enc_prenet.n_layers,
            config.model.enc_prenet.dropout,
        )
        dec_prenet_config = DecoderPrenetConfig(
            config.feature.n_mels,
            config.model.transformer.attention_dim,
            config.model.dec_prenet.n_units,
            config.model.dec_prenet.n_layers,
            config.model.dec_prenet.dropout,
        )
        postnet_config = PostnetConfig(
            config.feature.n_mels,
            config.model.postnet.conv_channels,
            config.model.postnet.kernel_size,
            config.model.postnet.n_layers,
            config.model.postnet.dropout,
        )

        self.encoder_input = nn.Sequential(
            EncoderPrenet(enc_prenet_config).to(device),
            ScaledPositionalEncoding(
                config.model.transformer.attention_dim,
                config.model.positional_encoding.dropout,
            ).to(device),
        )
        self.decoder_input = nn.Sequential(
            DecoderPrenet(dec_prenet_config).to(device),
            ScaledPositionalEncoding(
                config.model.transformer.attention_dim,
                config.model.positional_encoding.dropout,
            ).to(device),
        )
        self.transformer = nn.Transformer(
            config.model.transformer.attention_dim,
            config.model.transformer.nhead,
            config.model.transformer.num_encoder_layers,
            config.model.transformer.num_decoder_layers,
            config.model.transformer.dim_feedforward,
            config.model.transformer.dropout,
            batch_first=True,
            norm_first=config.model.transformer.norm_first,
            device=device,
        )
        self.postnet_proj = nn.Linear(
            config.model.transformer.attention_dim,
            config.feature.n_mels,
            bias=False,
            device=device,
        )
        self.postnet_residual = Postnet(postnet_config).to(device)
        self.embed_event_memory = None
        self.embed_event_decin = None
        if config.model.embed_event.memory is True:
            self.embed_event_memory = EventEmbedding(
                config.model.transformer.attention_dim + len(config.sound_event),
                config.model.transformer.attention_dim,
            ).to(device)
        if config.model.embed_event.decoder is True:
            self.embed_event_decin = EventEmbedding(
                config.model.transformer.attention_dim + len(config.sound_event),
                config.model.transformer.attention_dim,
            ).to(device)

    def encode(self, source, event_label, masks):
        """Return memory (encoded source).

        Args:
            source (Tensor) : onomatopoeia [batch_size, src_len, vocab_size]
            masks (dict): dictionary of mask tensors.

        Returns:
            memory (Tensor): encoded source tensor [batch_size, src_len, att_dim]
        """
        memory = self.transformer.encoder(
            self.encoder_input(source),
            masks["src_mask"],
            masks["src_key_padding_mask"],
        )
        if self.embed_event_memory is not None:
            memory = self.embed_event_memory(memory, event_label)
        return memory

    def decode(self, target, memory, event_label, masks):
        """Return decoded target tensor from memory and decoder inputs.

        Args:
            target (Tensor): log mel-spectrograms [batch_size, tgt_len, n_mels]
            memory (Tensor): encoder output    [batch_size, src_len, att_dim]
            masks (dict): dictionary of mask tensors.

        Returns:
            decoded (Tensor): decoded target [batch_size, tgt_len, att_dim]
        """
        decoder_input = self.decoder_input(target)
        if self.embed_event_decin is not None:
            decoder_input = self.embed_event_decin(decoder_input, event_label)
        decoded = self.transformer.decoder(
            decoder_input,
            memory,
            masks["tgt_mask"],
            masks["memory_mask"],
            masks["tgt_key_padding_mask"],
            masks["memory_key_padding_mask"],
        )
        return decoded

    def forward(self, source, target, event_label, masks):
        """Forward propagation.

        Args:
            source (Tensor) : onomatopoeia [batch_size, src_len, vocab_size]
            target (Tensor): log mel-spectrograms [batch_size, tgt_len, n_mels]
            masks (dict): dictionary of mask tensors.

        Returns:
            logmel (Tensor): log mel-spectrograms [batch_size, tgt_len, n_mels]
            refined (Tensor): log mel-spectrograms [batch_size, tgt_len, n_mels]
        """
        memory = self.encode(source, event_label, masks)
        decoded = self.decode(target, memory, event_label, masks)
        logmel = self.postnet_proj(decoded)
        refined = logmel + self.postnet_residual(logmel)
        return logmel, refined


class Mel2Linear(nn.Module):
    """Mel2Linear class.

    Convert log mel-spectrogram into log linear-spectrogram.

    Args:
        config (CBHGConfig):
            input_dim (int): number of convolution channels.
            n_convbanks (int): number of convolution banks.
            n_highways (int): number of layers in highway networks.
            projections (tuple): list of projection dims.
        device (torch.device): "cuda" or "cpu".
    """

    def __init__(self, config, device):
        """Initialize class."""
        super().__init__()
        cbhg_config = CBHGConfig(
            config.model.cbhg.proj_dim,
            config.model.cbhg.n_convbanks,
            config.model.cbhg.n_highways,
            (config.model.cbhg.proj_dim, config.feature.n_mels),
        )

        # 1x1 conv.
        self.pre_projection = nn.Conv1d(
            config.feature.n_mels,
            config.model.cbhg.proj_dim,
            kernel_size=1,
            device=device,
        )
        self.post_projection = nn.Conv1d(
            2 * config.model.cbhg.proj_dim,
            config.feature.n_fft // 2 + 1,
            kernel_size=1,
            device=device,
        )
        self.cbhg = CBHG(cbhg_config, device).to(device)

    def forward(self, inputs):
        """Forward propagation."""
        inputs = self.pre_projection(inputs.transpose(1, 2))
        outputs = self.cbhg(inputs.transpose(1, 2))
        outputs = self.post_projection(outputs.transpose(1, 2))
        return outputs.transpose(1, 2)


def get_model(cfg, phase, mapping_dict, device):
    """Instantiate models.

    Args:
        cfg (CBHGConfig) : configuration of models
        mapping_dict (MappingDict) : dictionaries between phonemes to ids.
    """
    if phase == "transformer":
        model = Seq2SeqTransformer(mapping_dict.char2id, cfg, device)
    else:
        model = Mel2Linear(cfg, device)

    return model
