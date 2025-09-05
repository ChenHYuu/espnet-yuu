from typing import Optional, Tuple, Union

import torch
import torchaudio as ta
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend

class SpectrumFrontend(AbsFrontend):

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        preemp: bool = True,
        power: int = 1,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        window_fn: str = "hamming",
        log: bool = False,
        normalize: Optional[str] = False,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.log = log
        self.preemp = preemp
        self.normalize = normalize
        if window_fn == "hann":
            self.window_fn = torch.hann_window
        elif window_fn == "hamming":
            self.window_fn = torch.hamming_window

        if preemp:
            self.register_buffer(
                "flipped_filter",
                torch.FloatTensor([-0.97, 1.0]).unsqueeze(0).unsqueeze(0),
            )

        self.transform = ta.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=self.window_fn,
            power=power,
            normalized=normalize
        )
    
    def output_size(self):
        return self.n_fft // 2 + 1 # bins of fft
        
    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if self.preemp:
                    # reflect padding to match lengths of in/out
                    x = input.unsqueeze(1)
                    x = F.pad(x, (1, 0), "reflect")

                    # apply preemphasis
                    x = F.conv1d(x, self.flipped_filter).squeeze(1)
                else:
                    x = input

                # apply frame feature extraction
                x = self.transform(x)

                if self.log:
                    x = torch.log(x + 1e-6)
                # if self.normalize is not None:
                #     if self.normalize == "mn":
                #         x = x - torch.mean(x, dim=-1, keepdim=True)
                #     else:
                #         raise NotImplementedError(
                #             f"got {self.normalize}, not implemented"
                #         )

        input_length = torch.Tensor([x.size(-1)]).repeat(x.size(0))

        return x.permute(0, 2, 1), input_length