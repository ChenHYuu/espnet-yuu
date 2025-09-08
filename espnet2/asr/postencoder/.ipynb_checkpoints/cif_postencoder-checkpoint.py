#!/usr/bin/env python3

"""CIF multi-task predictor postencoder"""

import torch
from typeguard import check_argument_types


from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


class CifPostencoder(AbsPostEncoder):

    def __init__(
        self,
        input_size: int,
        # input_layer: Optional[str] = None,
        output_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        # return_int_enc: bool = False,
    ):
        assert check_argument_types()
        super().__init__()