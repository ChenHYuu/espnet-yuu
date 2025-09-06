#!/usr/bin/env python3

"""CIF multi-task predictor postencoder"""

from typing import Optional, Tuple

import torch
from typeguard import check_argument_types


from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


class CifPostencoder(AbsPostEncoder):

    def __init__(
        self,
        input_size: int,
        l_order,
        r_order,
        # input_layer: Optional[str] = None,
        # output_size: Optional[int] = None,
        threshold = 1.0,
        dropout_rate: float = 0.1,
        tail_threshold=0.45,
        # return_int_enc: bool = False,
    ):
        assert check_argument_types()
        super().__init__()

        self.input_size = input_size

        self.pad = torch.nn.ConstantPad1d((l_order, r_order), 0)
        # Cif input_size = output_size
        self.cif_conv1d = torch.nn.Conv1d(input_size, input_size, l_order + r_order + 1,
                                          groups=input_size)
        self.cif_output = torch.nn.Linear(input_size, 1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.threshold = threshold

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor=None) -> Tuple[torch.Tensor]:
        
        h = input
        context = h.transpose(1, 2) # put time information inside
        queries = self.pad(context) # padding time information
        memory = self.cif_conv1d(queries)
        output = memory + context
        output = self.dropout(output)
        output = output.transpose(1, 2)
        output = torch.relu(output)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        # smoothing in future

        alphas = alphas.squeeze(-1)
        
        token_num = alphas.sum(-1)
        output_size = token_num.round().int()


        with torch.no_grad():
            threshold = token_num / token_num.ceil()
            # if self.training:
            #     threshold = token_num / token_num.ceil()
            # else:
            #     threshold = torch.ones(input.size(0), device=input.device)*self.threshold
        acoustic_embed = self.cif(input, alphas, threshold)

        return (acoustic_embed, token_num), output_size
        
    def cif(self, hidden, alphas, threshold=1.0):
        batch_size, len_time, hidden_size = hidden.size()

        device = hidden.device
        integrate = torch.zeros([batch_size], device=device)
        # buffer of one word
        frame = torch.zeros([batch_size, hidden_size], device=device)

        list_fires = []
        list_frames = []

        for t in range(len_time):
            alpha = alphas[:, t]
            # Remain how much to complete one word
            distribution_completion = torch.ones([batch_size], device=device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= threshold
            # If over than threshold, set integrate to reamin value or keep original value
            integrate = torch.where(fire_place, 
                                    integrate - torch.ones([batch_size], device=device),
                                    integrate)
            
            # If over than threshold, set to reamin value or is whole alpha
            # Current weight consider threshold
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            
            remains = alpha - cur
            
            # The information contributed by this frame
            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                                remains[:, None] * hidden[:, t, :],
                                frame)

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.nonzero(fire >= threshold[b]).squeeze())
            if max_label_len <= l.size(0):
                list_ls.append(l)
                continue
            pad_l = torch.zeros(max_label_len - l.size(0), hidden_size, device=device)
            list_ls.append(torch.cat([l, pad_l], 0))
        return torch.stack(list_ls, 0)
        
    def output_size(self) -> int:
        return self.input_size
