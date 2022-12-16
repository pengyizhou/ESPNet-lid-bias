# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any, List, Sequence, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface


class BaseTransformerDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        lid_vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        hidden_layer_idx: List[int] = [],
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        concat_dim = attention_dim+lid_vocab_size
        self.lid_vocab_size = lid_vocab_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")


        self.proj_concat = torch.nn.Linear(concat_dim, attention_dim)
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm_asr = LayerNorm(attention_dim)
            self.after_norm_lid = LayerNorm(attention_dim)

        self.output_layer_asr = torch.nn.Linear(attention_dim, vocab_size)
        self.output_layer_lid = torch.nn.Linear(attention_dim, lid_vocab_size)
        
        # Must set by the inheritance
        self.decoders_asr = None
        self.decoders_lid = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )

        x = self.embed(tgt)

        x_lid, tgt_mask_lid, memory_lid, memory_mask_lid = self.decoders_lid(x, tgt_mask, memory, memory_mask)
        x_lid = self.after_norm_lid(x_lid)
        x_lid = self.output_layer_lid(x_lid)
        lid_post = torch.nn.functional.softmax(x_lid.detach().clone(), dim=-1)
        sos_lid = torch.zeros((lid_post.size(0),1,self.lid_vocab_size))
        sos_lid[:,:,-1] = 1.0
        sos_lid = sos_lid.type_as(lid_post)
        lid_post = torch.concat((sos_lid, lid_post[:,:-1,:]),dim=1).type_as(x)
        x = torch.cat((lid_post, x), dim=-1)
        x = self.proj_concat(x)
        x_asr, tgt_mask_asr, memory_asr, memory_mask_asr = self.decoders_asr(x, tgt_mask, memory, memory_mask)

        if self.normalize_before:
            x_asr = self.after_norm_asr(x_asr)
        if self.output_layer_asr is not None and self.output_layer_lid is not None:
            x_asr = self.output_layer_asr(x_asr)

        olens = tgt_mask_asr.sum(1)

        return x_asr, x_lid, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        lid_post: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = torch.cat((lid_post, tgt), dim=-1)
        x = self.proj_concat(x)
        if cache is None:
            cache = [None] * (len(self.decoders_asr))
        new_cache = []

        for c, decoder in zip(cache, self.decoders_asr):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm_asr(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer_asr is not None:
            y = torch.log_softmax(self.output_layer_asr(y), dim=-1)

        return y, new_cache

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders_asr)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        x = self.embed(ys)
        x_lid, tgt_mask_lid, memory_lid, memory_mask_lid = self.decoders_lid(x, None, xs, None)
        x_lid = self.after_norm_lid(x_lid)
        x_lid = self.output_layer_lid(x_lid)
        lid_post = torch.nn.functional.softmax(x_lid.detach().clone(), dim=-1)
        sos_lid = torch.zeros((lid_post.size(0),1,self.lid_vocab_size))
        sos_lid[:,:,-1] = 1.0
        sos_lid = sos_lid.type_as(lid_post)
        lid_post = torch.concat((sos_lid, lid_post[:,:-1,:]),dim=1).type_as(x)

        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(x, ys_mask, xs, lid_post=lid_post, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list


class TransformerDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        lid_vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        hidden_layer_idx: List[int] = [],
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            lid_vocab_size=lid_vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            hidden_layer_idx=hidden_layer_idx,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        # number_asr = num_blocks - number_shared
        # number_lid = number_asr
        # self.decoders_share = None
        self.decoders_asr = None
        self.decoders_lid = None

        # self.decoders_share = repeat(
        #     number_shared,
        #     lambda lnum: DecoderLayer(
        #         attention_dim,
        #         MultiHeadedAttention(
        #             attention_heads, attention_dim, self_attention_dropout_rate
        #         ),
        #         MultiHeadedAttention(
        #             attention_heads, attention_dim, src_attention_dropout_rate
        #         ),
        #         PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
        #         dropout_rate,
        #         normalize_before,
        #         concat_after,
        #     ),
        # )
        self.decoders_asr = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.decoders_lid = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

