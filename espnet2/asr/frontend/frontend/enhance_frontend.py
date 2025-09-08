import copy
import math
from typing import Optional, OrderedDict, Tuple, Union
import librosa

import humanfriendly
import numpy as np
import torch
from torch import nn
from torch_complex.tensor import ComplexTensor
from minGRU_pytorch import minGRU
from typeguard import typechecked

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)

from espnet.nets.pytorch_backend.transformer.encoder_mix import EncoderMix
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

class EnhanceFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Log-Mel-Fbank
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        type = 'frame_mask_v1',
        dropout_rate = 0.0,
        add_awgn = False,
        snr = 20,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length

        self.n_fft = n_fft

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        self.add_awgn = add_awgn
        self.snr = snr
        self.dropout_rate = dropout_rate

        if type == 'frame_mask_v1':
            self.build_ehance_model_fmv1()
            self.enhance = self.enhance_fmv1

        elif type == 'frame_mask_v2':
            self.build_ehance_model_fmv2()
            self.enhance = self.enhance_fmv2
        
        elif type == 'frame_mask_v3':
            self.build_ehance_model_fmv3()
            self.enhance = self.enhance_fmv3

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

    def build_ehance_model_fmv1(self):
        self.cnn_encoder = torch.nn.Sequential(
            Conv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            Conv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            Conv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            Conv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
        )

        self.d_feats = self.n_fft // 2 + 1
        ffn_size = self.d_feats
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = ffn_size * 128

        self.ffn_size = ffn_size

        self.output_ffn = nn.Sequential(
            nn.Linear(ffn_size, self.d_feats),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_feats, 1),
        )
        self.output_layer = nn.Softplus()

    def enhance_fmv1(self, input, input_len):
        input_feats = nn.functional.pad(input, (0, 0, 8, 0))
        input_feats = torch.unsqueeze(input_feats, dim=1)
        input_feats = self.cnn_encoder(input_feats)

        B, C, K, D = input_feats.shape
        input_feats = torch.permute(input_feats, (0, 2, 1, 3))
        input_feats = torch.reshape(input_feats, (B, K, C*D))

        input_feats = self.output_ffn(input_feats)
        input_feats = self.output_layer(input_feats)
        input_feats = input_feats * input

        return input_feats, None

    def build_ehance_model_fmv2(self):
        self.cnn_encoder = nn.Sequential(
            Conv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            Conv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            Conv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            Conv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
        )

        self.d_feats = self.n_fft // 2 + 1
        ffn_size = self.d_feats
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = (ffn_size - (3 - 1) - 1 ) // 2 + 1
        ffn_size = ffn_size * 128

        hidden_size = 64
        #self.hidden_size = hidden_size
        #self.rnn = nn.GRU(ffn_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=self.dropout_rate)
        #self.rnn_input = nn.Linear(ffn_size, hidden_size)
        #self.rnn = minGRU(hidden_size)

        self.output_ffn = nn.Sequential(
            nn.Linear(ffn_size, hidden_size),
            #nn.ELU(),
	        minGRU(hidden_size),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size, 1),
        )
        self.output_layer = nn.Softplus()

    def enhance_fmv2(self, input, input_lens):
        input_feats = nn.functional.pad(input, (0, 0, 8, 0))
        input_feats = torch.unsqueeze(input_feats, dim=1)
        input_feats = self.cnn_encoder(input_feats)

        B, C, K, D = input_feats.shape
        input_feats = torch.permute(input_feats, (0, 2, 1, 3))
        input_feats = torch.reshape(input_feats, (B, K, C*D))

        # RNN
        device = input.device
        #with torch.autocast(device_type='cuda', dtype=torch.float32):
            #h0 = torch.zeros(1, B, self.hidden_size, device=device)
            #rnn_feats, _ = self.rnn(input_feats, h0)
            # rnn_feats = self.rnn_input(input_feats)
            # rnn_feats = self.rnn(rnn_feats)

        input_feats = self.output_ffn(input_feats)
        input_feats = self.output_layer(input_feats)
        # The scale needs to power since we want do scale on
        # magnitude not power spectrum
        input_feats = input_feats * input

        return input_feats, None

    def build_ehance_model_fmv3(self):
        input_dim = self.n_fft // 2 + 1

        # (voice, burst, noise, silence)
        scale_table = torch.Tensor([1, 0.3, 0, 0])
        self.register_buffer("scale_table", scale_table)

        self.input_ffn = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 80)
        )

        self.looking_size = 3
        self.cnn_layers = 2
        self.dropout = 0.1
        self.cnn_encoder = nn.Sequential(
            Conv2dCell(in_channels=1, out_channels=8, kernel_size=(self.looking_size, 5), stride=(1, 2), activation=nn.ReLU(), dropout=self.dropout),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
            nn.Dropout(self.dropout),
            Conv2dCell(in_channels=8, out_channels=16, kernel_size=(self.looking_size, 5), stride=(1, 2), activation=nn.ReLU(), dropout=self.dropout),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2)),
            nn.Dropout(self.dropout),
        )

        ffn_size = 80
        ffn_size = ((ffn_size - 1) // 2) // 2
        ffn_size = ((ffn_size - 1) // 2) // 2
        ffn_size = ffn_size * 16

        self.ffn_size = ffn_size
        self.linear1 = nn.Linear(ffn_size, 257)
        self.dropout = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(257, 4)

    def enhance_fmv3(self, input, input_len):
        input_feats = self.input_ffn(input)

        input_feats = torch.unsqueeze(input_feats, dim=1)
        input_feats = nn.functional.pad(input_feats, (0, 0, (self.looking_size)*self.cnn_layers, 0))
        input_feats = self.cnn_encoder(input_feats)

        B, C, K, D = input_feats.shape
        input_feats = torch.permute(input_feats, (0, 2, 1, 3))
        input_feats = torch.reshape(input_feats, (B, K, C*D))

        input_feats = self.dropout(self.linear1(input_feats))
        input_feats = nn.functional.gumbel_softmax(self.linear2(input_feats))
        input_feats = torch.matmul(input_feats, self.scale_table).unsqueeze(dim=-1)
        input_feats = input_feats ** 2 
        input_feats = input_feats * input

        return input_feats, None

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # torch.autograd.set_detect_anomaly(True)
        if self.add_awgn and self.training:
            # alpha = input.abs().max(dim=-1, keepdim=True).values * 0.1
            # awgn1 = alpha * torch.randn(input.shape, device=device)
            # awgn2 = alpha * torch.randn(input.shape, device=device)
            # awgn3 = alpha * torch.randn(input.shape, device=device)
            input = add_awgn_batch(input, self.snr)
    
        input_stft, feats_lens = self._compute_stft(input, input_lengths)
        input_magn = torch.sqrt(input_stft.real ** 2 + input_stft.imag ** 2)
        enhanced_magn, _ = self.enhance(input_magn, feats_lens)
        enhanced_wav, enhanced_wav_lens = self._to_wavform(enhanced_magn, input_stft, input_lengths)


        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(enhanced_wav, enhanced_wav_lens)
        else:
            input_stft = ComplexTensor(enhanced_wav[..., 0], enhanced_wav[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        return input_feats, feats_lens
    
    def _to_wavform(self, input_magn, input_stft, input_lens):
        input_phase = torch.complex(input_stft.real, input_stft.imag)
        input_phase = torch.angle(input_phase)

        # r = torch.sqrt(input_power) # convert power spectrum to magnitude spectrum
        cos = torch.cos(input_phase)
        sin = torch.sin(input_phase)
        norm = torch.complex(cos, sin)
        
        spec_with_phase = input_magn * norm
        wav, wav_lens = self.stft.inverse(spec_with_phase, input_lens)
        wav = nn.functional.tanh(wav)
        # wav = self._unitize(wav)

        return wav, wav_lens

    def _unitize(self, input: torch.Tensor):       
        max = input.abs.max(dim=1, keepdim=True).values
        min = -max
        # min = input.min(dim=1, keepdim=True).values
        # max = input.max(dim=1, keepdim=True).values

        range_values = max - min
        range_values[range_values == 0] = 1e-8  # Replace zeros with a small value

        unitized_data = ((input - min) / range_values) * 2 -1

        return unitized_data

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens



class IRMEnhanceFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Log-Mel-Fbank
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,# Create a mask using tensor operations
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        type = 'irm_v1',
        hb_start=100,
        hb_end=300,
        dropout_rate = 0.0,
        add_awgn=False,
        snr=20,
        return_spec=False,
    ):
        # super().__init__(fs, n_fft, win_length, hop_length, window, center, normalized,
        #                  onesided, n_mels, fmin, fmax, htk, frontend_conf, apply_stft)
        super().__init__()
        
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length

        self.fs = fs
        self.n_fft = n_fft
        self.window = window
        self.dropout_rate = dropout_rate
        self.return_spec = return_spec

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.add_awgn = add_awgn
        self.snr = snr
        self.hb_start = hb_start
        self.hb_end = hb_end

        if type == 'irm_v1':
            self.build_ehance_model_irmv1()
            self.enhance = self.enhance_irmv1
        elif type == 'irm_v2':
            self.generate_harmonic_bank()
            self.build_ehance_model_irmv2()
            self.enhance = self.enhance_irmv2
        
        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

    def norm(self, data):
        max = data.max()
        min = data.min()
        dis = max - min
        data = data - min
        data = data / dis
        data = data * 2 - 1
        return data

    def build_ehance_model_irmv1(self):
        input_dim = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,

        # Convolution 2d encoder
        self.encoder = torch.nn.ModuleDict({
            'conv2d_1': Conv2dCell(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_2': Conv2dCell(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_3': Conv2dCell(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_4': Conv2dCell(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_5': Conv2dCell(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
        })
        # RNN 
        self.hidden_size = 128
        self.d_feats = self.n_fft // 2 + 1
        rnn_input_size = self.d_feats
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = rnn_input_size * 256
        self.rnn_input_size = rnn_input_size
        self.rnn = torch.nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout_rate)
        self.linear = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.rnn_input_size),
            nn.Dropout(self.dropout_rate),
        )
    
        # De-convolution 2d decoder
        self.decoder = torch.nn.ModuleDict({
            'deconv2d_1': ConvTranspose2dCell(in_channels=512, out_channels=128, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_2': ConvTranspose2dCell(in_channels=256, out_channels=64, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_3': ConvTranspose2dCell(in_channels=128, out_channels=32, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_4': ConvTranspose2dCell(in_channels=64, out_channels=16, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_5': ConvTranspose2dCell(in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2), activation=nn.Softplus(), dropout=self.dropout_rate),
        })
    
    def enhance_irmv1(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        input = torch.nn.functional.pad(input, (0, 0, 5, 0))
        # input = torch.sqrt(input) # power to magnitude
        # input = torch.log10(input + 0.00001)
        input_feats = torch.unsqueeze(input, dim=1)

        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        encoder_temp = [0]*5
        i = 0
        for k, conv in self.encoder.items():
            if i == 0:
                encoder_temp[i] = conv(input_feats)
            else:
                encoder_temp[i] = conv(encoder_temp[i-1])
            i += 1
        encode_feats = encoder_temp[i-1]

        # reshape: flatten along with channels
        B, C, K, D = encode_feats.shape
        encode_feats = torch.permute(encode_feats, (0, 2, 1, 3))
        encode_feats = torch.reshape(encode_feats, (B, K, C*D))

        # RNN
        device = encode_feats.device
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            h0 = torch.zeros(2, B, self.hidden_size, device=device)
            # c0 = torch.zeros(2, B, self.hidden_size, device=device)
            rnn_feats, _ = self.rnn(encode_feats, h0)
            rnn_feats = self.linear(rnn_feats)

        # reshape
        decoder_input = torch.reshape(rnn_feats, (B, K, C, D))
        decoder_input = torch.permute(decoder_input, (0, 2, 1, 3))   
        
        i = 4
        for k, deconv in self.decoder.items():
            if decoder_input.shape[-1] != encoder_temp[i].shape[-1]:
                decoder_input = torch.nn.functional.pad(
                    decoder_input,
                    (0, 1, 0, 0)
                )
            decoder_input = torch.concat((decoder_input, encoder_temp[i]), dim=1)
            decoder_input = deconv(decoder_input)
            i -= 1
        enhanced_feats = torch.squeeze(decoder_input, dim=1)
        # enhanced_feats = enhanced_feats ** 2
        enhanced_feats = input * enhanced_feats
        # enhanced_feats = enhanced_feats ** 2
        enhanced_feats = enhanced_feats[:, 5:, :]

        return enhanced_feats, None, None

    def build_ehance_model_irmv2(self):
        input_dim = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,

        # Convolution 2d encoder
        self.encoder = torch.nn.ModuleDict({
            'conv2d_1': Conv2dCell(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_2': Conv2dCell(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_3': Conv2dCell(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_4': Conv2dCell(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_5': Conv2dCell(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
        })
        # RNN 
        self.hidden_size = self.K
        self.d_feats = self.n_fft // 2 + 1
        rnn_input_size = self.d_feats
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = rnn_input_size * 256
        self.rnn_input_size = rnn_input_size
        self.rnn = torch.nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout_rate)
        self.linear = nn.Sequential(
            nn.Linear(self.d_feats + self.hidden_size, self.rnn_input_size),
        )
    
        # De-convolution 2d decoder
        self.decoder = torch.nn.ModuleDict({
            'deconv2d_1': ConvTranspose2dCell(in_channels=512, out_channels=128, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_2': ConvTranspose2dCell(in_channels=256, out_channels=64, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_3': ConvTranspose2dCell(in_channels=128, out_channels=32, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_4': ConvTranspose2dCell(in_channels=64, out_channels=16, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_5': ConvTranspose2dCell(in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2), activation=nn.Softplus(), dropout=self.dropout_rate),
        })

    def enhance_irmv2(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        input = torch.nn.functional.pad(input, (0, 0, 5, 0))
        # input = torch.sqrt(input) # power to magnitude
        # input = torch.log10(input + 0.00001)
        input_feats = torch.unsqueeze(input, dim=1)

        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        encoder_temp = [0]*5
        i = 0
        for k, conv in self.encoder.items():
            if i == 0:
                encoder_temp[i] = conv(input_feats)
            else:
                encoder_temp[i] = conv(encoder_temp[i-1])
            i += 1
        encode_feats = encoder_temp[i-1]

        # reshape: flatten along with channels
        B, C, K, D = encode_feats.shape
        encode_feats = torch.permute(encode_feats, (0, 2, 1, 3))
        encode_feats = torch.reshape(encode_feats, (B, K, C*D))

        # RNN
        device = encode_feats.device
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            h0 = torch.zeros(2, B, self.hidden_size, device=device)
            # c0 = torch.zeros(2, B, self.hidden_size, device=device)
            rnn_feats, _ = self.rnn(encode_feats, h0)
            residual = rnn_feats
            hb_score = nn.functional.gumbel_softmax(rnn_feats, 1.0, hard=True, dim=-1)
            rnn_feats = torch.matmul(hb_score, self.harmonic_bank)
            rnn_feats = torch.concat((rnn_feats, residual), dim=-1)
            rnn_feats = self.linear(rnn_feats)

        # reshape
        decoder_input = torch.reshape(rnn_feats, (B, K, C, D))
        decoder_input = torch.permute(decoder_input, (0, 2, 1, 3))   
        
        i = 4
        for k, deconv in self.decoder.items():
            if decoder_input.shape[-1] != encoder_temp[i].shape[-1]:
                decoder_input = torch.nn.functional.pad(
                    decoder_input,
                    (0, 1, 0, 0)
                )
            decoder_input = torch.concat((decoder_input, encoder_temp[i]), dim=1)
            decoder_input = deconv(decoder_input)
            i -= 1
        enhanced_feats = torch.squeeze(decoder_input, dim=1)
        # enhanced_feats = enhanced_feats ** 2
        enhanced_feats = input * enhanced_feats
        # enhanced_feats = enhanced_feats ** 2
        enhanced_feats = enhanced_feats[:, 5:, :]

        return enhanced_feats, None, None

    def generate_harmonic_bank(self, half=False):
        n_fft = self.n_fft
        fs = self.fs
        # up_freq = self.up_freq
        if half == True:
            d_feats = n_fft // 4 + 1
        else:
            d_feats = n_fft // 2 + 1
        
        freq_max = 4000
        times = 100
        length = 16000

        # Convert freq to tone scale
        freq_start = self.hb_start
        freq_end = self.hb_end
        tone_scale = 24
        x_start = math.ceil(tone_scale * math.log2(freq_start))
        x_end = math.floor(tone_scale * math.log2(freq_end))
        freq_bins = [freq for freq in range(x_start, x_end+1, 1)]
        K = len(freq_bins) + 1

        harmonic_bank = np.zeros([K, d_feats], dtype=np.float32)
        sample_pos = int((16000 / self.hop_length + 1 ) / 2)
        count = 1
        for bin in freq_bins:
            # Reverse to freq
            i = round(math.pow(2, bin/tone_scale))
            tmp_freq_max = freq_max if i * times > freq_max else i * times
            tmp_freq = [j for j in range(i, tmp_freq_max+1, i)]
            # num_freq = len(tmp_freq)
            sine = np.zeros(16000)
            for freq in tmp_freq:
                sine += librosa.tone(freq, sr=16000, length=length)
            sine = self.norm(sine)
            # sine = sine / num_freq
            S = librosa.stft(y=sine, 
                             n_fft=self.n_fft, 
                             win_length=self.n_fft, 
                             hop_length=self.hop_length, 
                             window=self.window,
                             pad_mode='reflect')
            magn = np.sqrt(S.real**2 + S.imag ** 2)
            magn = magn.transpose()
            # power /= num_freq
            # print(power)
            harmonic_bank[count, :] = magn[sample_pos, :d_feats]
            # power[sample_pos, :] = np.log10(power[sample_pos, :] + 0.1)
            # power[sample_pos, :] = power[sample_pos, :] / np.sum(np.absolute(power[sample_pos, :]))
            # harmonic_bank[count, :] = power[sample_pos, :]
            count += 1
        self.K = K
        harmonic_bank = torch.from_numpy(harmonic_bank)
        # harmonic_bank = torch.sqrt(harmonic_bank)
        # harmonic_bank = torch.log10(harmonic_bank + 1e-10)
        self.register_buffer("harmonic_bank", harmonic_bank)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # torch.autograd.set_detect_anomaly(True)
        input_power_clean = None
        if self.return_spec:
            input_stft_clean, _ = self._compute_stft(input, input_lengths)
            input_power_clean = input_stft_clean.real ** 2 + input_stft_clean.imag ** 2

        if self.add_awgn and self.training:
            # alpha = input.abs().max(dim=-1, keepdim=True).values * 0.1
            # awgn1 = alpha * torch.randn(input.shape, device=device)
            # awgn2 = alpha * torch.randn(input.shape, device=device)
            # awgn3 = alpha * torch.randn(input.shape, device=device)
            input = add_awgn_batch(input, self.snr)

        input_stft, feats_lens = self._compute_stft(input, input_lengths)
        input_magn = torch.sqrt(input_stft.real ** 2 + input_stft.imag ** 2)
        enhanced_magn, _, _ = self.enhance(input_magn)
        enhanced_wav, enhanced_wav_lens = self._to_wavform(enhanced_magn, input_stft, input_lengths)

        feats_intermediate = (input_power_clean, enhanced_magn ** 2)

        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(enhanced_wav, enhanced_wav_lens)
        else:
            input_stft = ComplexTensor(enhanced_wav[..., 0], enhanced_wav[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        if self.return_spec:
            return input_feats, feats_lens, feats_intermediate
        else:
            return input_feats, feats_lens

    def _to_wavform(self, input_magn, input_stft, input_lens):
        input_phase = torch.complex(input_stft.real, input_stft.imag)
        input_phase = torch.angle(input_phase)

        # r = torch.sqrt(input_power) # convert power spectrum to magnitude spectrum
        cos = torch.cos(input_phase)
        sin = torch.sin(input_phase)
        norm = torch.complex(cos, sin)
        
        spec_with_phase = input_magn * norm
        wav, wav_lens = self.stft.inverse(spec_with_phase, input_lens)
        # wav = self._unitize(wav)
        wav = nn.functional.tanh(wav)

        return wav, wav_lens

    def _unitize(self, input: torch.Tensor):       
        max = input.abs.max(dim=1, keepdim=True).values
        min = -max
        # min = input.min(dim=1, keepdim=True).values
        # max = input.max(dim=1, keepdim=True).values

        range_values = max - min
        range_values[range_values == 0] = 1e-8  # Replace zeros with a small value

        unitized_data = ((input - min) / range_values) * 2 -1

        return unitized_data

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens 
    
    def output_size(self) -> int:
        return self.n_mels

class ConversionFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Log-Mel-Fbank
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,# Create a mask using tensor operations
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        type = 'vc_v1',
        dropout_rate = 0.0,
        use_gumbel=True,
        add_awgn=False,
        snr=20,
        hb_start=100,
        hb_end=300,
        hb_att_layer=2,
        hb_att_norm=False,
        irm_source=False,
    ):
        # super().__init__(fs, n_fft, win_length, hop_length, window, center, normalized,
        #                  onesided, n_mels, fmin, fmax, htk, frontend_conf, apply_stft)
        super().__init__()

        self.return_spec = False
        
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length

        self.fs = fs
        self.n_fft = n_fft
        self.window = window
        self.dropout_rate = dropout_rate
        self.use_gumbel = use_gumbel

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.add_awgn = add_awgn
        self.snr = snr
        self.hb_start = hb_start
        self.hb_end = hb_end
        self.hb_att_layer = hb_att_layer
        self.hb_att_norm = hb_att_norm

        self.irm_source = irm_source
        if irm_source:
            self.irm = IRMBlock(n_fft=n_fft, dropout_rate=0.0)

        if type == 'vc_v1':
            self.generate_harmonic_bank(half=True)
            self.build_ehance_model_vc1()
            self.enhance = self.enhance_vc1
        elif type == 'vc_v2':
            self.generate_harmonic_bank(half=True)
            self.build_ehance_model_vc2()
            self.enhance = self.enhance_vc2
        elif type == 'vc_v3':
            self.generate_harmonic_bank(half=True)
            self.build_ehance_model_vc3()
            self.enhance = self.enhance_vc3
        elif type == 'vc_v4':
            self.generate_harmonic_bank(half=True)
            self.build_ehance_model_vc4()
            self.enhance = self.enhance_vc4

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

    def build_ehance_model_vc1(self):
        self.d_feats = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,
        self.input_norm = nn.BatchNorm1d(self.d_feats)

        # Convolution 2d encoder
        self.high_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_5': GatedConv2dCell(in_channels=128, out_channels=256, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        self.low_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(6, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(6, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(6, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            # 'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(5, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            # 'conv2d_5': GatedConv2dCell(in_channels=128, out_channels=256, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        # RNN 
        self.hidden_size = 128

        high_rnn_size = self.d_feats // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = high_rnn_size * 256
        # self.linear_high_in = nn.Linear(high_rnn_size, self.hidden_size)
        # self.high_rnn = minGRU(dim=self.hidden_size)
        # # self.high_rnn = nn.GRU(input_size=high_rnn_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True, dropout=self.dropout_rate)
        # self.linear_high_out = nn.Sequential(
        #     torch.nn.Linear(self.hidden_size, high_rnn_size),
        #     nn.Dropout(self.dropout_rate),
        # )
        high_att_list = []
        for i in range(self.hb_att_layer):
            high_att_list.append(
                GumbelMultiHeadedAttention(
                    1, high_rnn_size, self.d_feats // 2 + 1, self.K, high_rnn_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
            if self.hb_att_norm:
                high_att_list.append(nn.LayerNorm(high_rnn_size))
        self.high_attention = nn.ModuleList(high_att_list)


        low_rnn_size = self.d_feats // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = low_rnn_size * 64
        
        low_att_list = []
        for i in range(self.hb_att_layer):
            low_att_list.append(
                GumbelMultiHeadedAttention(
                    1, low_rnn_size, self.d_feats // 2 + 1, self.K, low_rnn_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
            if self.hb_att_norm:
                low_att_list.append(nn.LayerNorm(low_rnn_size))
        self.low_attention = nn.ModuleList(low_att_list)

       
        # De-convolution 2d decoder
        self.low_decoder = torch.nn.ModuleDict({
            # 'deconv2d_1': GatedConvTranspose2dCell(in_channels=256, out_channels=128, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            # 'deconv2d_1': GatedConvTranspose2dCell(in_channels=128, out_channels=64, kernel_size=(5, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_1': GatedConvTranspose2dCell(in_channels=64, out_channels=32, kernel_size=(6, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_2': GatedConvTranspose2dCell(in_channels=32, out_channels=16, kernel_size=(6, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_3': GatedConvTranspose2dCell(in_channels=16, out_channels=1, kernel_size=(6, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        self.high_decoder = torch.nn.ModuleDict({
            'deconv2d_1': GatedConvTranspose2dCell(in_channels=256, out_channels=128, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_2': GatedConvTranspose2dCell(in_channels=128, out_channels=64, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_3': GatedConvTranspose2dCell(in_channels=64, out_channels=32, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_4': GatedConvTranspose2dCell(in_channels=32, out_channels=16, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_5': GatedConvTranspose2dCell(in_channels=16, out_channels=1, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Linear(258, 257)
        )

    def enhance_vc1(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        # input = torch.sqrt(input)
        input = torch.nn.functional.pad(input, (0, 0, 16, 0))
        input = torch.log10(input + 1e-10)

        # input normalization
        input = torch.permute(input, (0, 2, 1))
        input = self.input_norm(input)
        input = torch.permute(input, (0, 2, 1))

        B, K, D = input.shape
        input = torch.unsqueeze(input, dim=1)
        hb_feats = torch.unsqueeze(self.harmonic_bank, dim=0).repeat(B, 1, 1)
        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        # Chunk along feats
        input_low, input_high = torch.split(input, [129, 128], dim=-1)
        input_high = torch.nn.functional.pad(input_high, (0, 1, 0, 0))

        encoder_temp_low = [0]*len(self.low_encoder)
        i = 0 
        for k, conv in self.low_encoder.items():
            if i == 0:
                encoder_temp_low[i] = conv(input_low)
            else:
                encoder_temp_low[i] = conv(encoder_temp_low[i-1])
            i += 1
        encode_low_feats = encoder_temp_low[i-1]

        # Middle Layers
        # reshape: flatten along with channels
        B, C, K, D = encode_low_feats.shape
        encode_low_feats = torch.permute(encode_low_feats, (0, 2, 1, 3))
        encode_low_feats = torch.reshape(encode_low_feats, (B, K, C*D))

        device = encode_low_feats.device
        # h0 = torch.zeros(1, B, self.hidden_size, device=device)
        # rnn_low_feats, _ = self.low_rnn(encode_low_feats, h0)
        
        att_feats = encode_low_feats
        residual = encode_low_feats
        for i in range(len(self.low_attention)):
            if self.hb_att_norm:
                if isinstance(self.low_attention[i], MultiHeadedAttention):
                    att_feats = self.low_attention[i](att_feats, hb_feats, hb_feats, None)
                elif isinstance(self.low_attention[i], nn.LayerNorm): # layernorm
                    att_feats = self.low_attention[i](att_feats + residual)
                    residual = att_feats
            else:
                att_feats = self.low_attention[i](att_feats, hb_feats, hb_feats, None)
                att_feats = att_feats + residual
                residual = att_feats
        # att_feats = self.attention[-1](att_feats, hb_feats, hb_feats, None)


        # att_feats = self.linear_low(att_feats)
        
        # reshape
        decoder_low_input = torch.reshape(att_feats, (B, K, C, D))
        decoder_low_input = torch.permute(decoder_low_input, (0, 2, 1, 3))   
        
        i = len(self.low_encoder)-1
        for k, deconv in self.low_decoder.items():
            if decoder_low_input.shape[-1] < encoder_temp_low[i].shape[-1]:
                decoder_low_input = torch.nn.functional.pad(
                    decoder_low_input,
                    (0, encoder_temp_low[i].shape[-1] - decoder_low_input.shape[-1], 0, 0)
                )
            # print(decoder_input.shape[-1], encoder_temp[i].shape[-1])
            # decoder_low_input = torch.concat((decoder_low_input, encoder_temp_low[i]), dim=1)
            decoder_low_input = deconv(decoder_low_input)
            i -= 1

        encoder_temp_high = [0]*len(self.high_encoder)
        i = 0 
        for k, conv in self.high_encoder.items():
            if i == 0:
                encoder_temp_high[i] = conv(input_high)
            else:
                encoder_temp_high[i] = conv(encoder_temp_high[i-1])
            i += 1
        encode_high_feats = encoder_temp_high[i-1]

        B, C, K, D = encode_high_feats.shape
        encode_high_feats = torch.permute(encode_high_feats, (0, 2, 1, 3))
        encode_high_feats = torch.reshape(encode_high_feats, (B, K, C*D))

        # h0 = torch.zeros(1, B, self.hidden_size, device=device)
        # encode_high_feats = self.linear_high_in(encode_high_feats)
        # rnn_high_feats = self.high_rnn(encode_high_feats)
        # rnn_high_feats = self.linear_high_out(rnn_high_feats)
        att_feats = encode_high_feats
        residual = encode_high_feats
        for i in range(len(self.high_attention)):
            if self.hb_att_norm:
                if isinstance(self.high_attention[i], MultiHeadedAttention):
                    att_feats = self.high_attention[i](att_feats, hb_feats, hb_feats, None)
                elif isinstance(self.high_attention[i], nn.LayerNorm): # layernorm
                    att_feats = self.high_attention[i](att_feats + residual)
                    residual = att_feats
            else:
                att_feats = self.high_attention[i](att_feats, hb_feats, hb_feats, None)
                att_feats = att_feats + residual
                residual = att_feats

        # decoder_high_input = torch.reshape(rnn_high_feats, (B, K, C, D))
        decoder_high_input = torch.reshape(att_feats, (B, K, C, D))
        decoder_high_input = torch.permute(decoder_high_input, (0, 2, 1, 3))   
        
        i = len(self.high_encoder) - 1
        for k, deconv in self.high_decoder.items():
            if decoder_high_input.shape[-1] < encoder_temp_high[i].shape[-1]:
                decoder_high_input = torch.nn.functional.pad(
                    decoder_high_input,
                    (0, encoder_temp_high[i].shape[-1] - decoder_high_input.shape[-1], 0, 0)
                )
            # print(decoder_input.shape[-1], encoder_temp[i].shape[-1])
            # decoder_high_input = torch.concat((decoder_high_input, encoder_temp_high[i]), dim=1)
            decoder_high_input = deconv(decoder_high_input)
            i -= 1
        
        enhanced_feats = torch.concat([decoder_low_input, decoder_high_input], dim=-1)
        # enhanced_feats = enhanced_feats[:, :, :, :-1]
        enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = torch.squeeze(enhanced_feats, dim=1)
        # enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = enhanced_feats[:, 16:, :]
        enhanced_feats = 10 ** enhanced_feats
        # enhanced_feats = enhanced_feats ** 2 # magnitude to power

        return enhanced_feats, None, None

    def build_ehance_model_vc2(self):
        self.d_feats = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,
        self.input_norm = nn.BatchNorm1d(self.d_feats)

        # Convolution 2d encoder
        self.high_encoder = torch.nn.ModuleDict({
            'conv2d_1': Conv2dCell(in_channels=1, out_channels=16, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_2': Conv2dCell(in_channels=16, out_channels=32, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_3': Conv2dCell(in_channels=32, out_channels=64, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_4': Conv2dCell(in_channels=64, out_channels=128, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
        })

        self.low_encoder = torch.nn.ModuleDict({
            'conv2d_1': Conv2dCell(in_channels=1, out_channels=16, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_2': Conv2dCell(in_channels=16, out_channels=32, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_3': Conv2dCell(in_channels=32, out_channels=64, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_4': Conv2dCell(in_channels=64, out_channels=128, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
        })

        # RNN 
        self.hidden_size = 128

        high_rnn_size = self.d_feats // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = high_rnn_size * 128
        self.linear_high_in = nn.Linear(high_rnn_size, self.hidden_size)
        self.high_rnn = minGRU(dim=self.hidden_size)
        self.linear_high_out = nn.Sequential(
            torch.nn.Linear(self.hidden_size, high_rnn_size),
            nn.Dropout(self.dropout_rate),
        )

        low_rnn_size = self.d_feats // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = low_rnn_size * 128
        
        self.attention = nn.ModuleList([
            GumbelMultiHeadedAttention(
                1, low_rnn_size, self.d_feats // 2 + 1, self.K, low_rnn_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
            ),
            # nn.LayerNorm(low_rnn_size),
            GumbelMultiHeadedAttention(
                1, low_rnn_size, self.d_feats // 2 + 1, self.K, low_rnn_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
            ),
            # nn.LayerNorm(low_rnn_size),
            # GumbelMultiHeadedAttention(
            #     1, low_rnn_size, self.d_feats // 2 + 1, self.K, low_rnn_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
            # ),
            # nn.LayerNorm(low_rnn_size),
        ])
       
        # De-convolution 2d decoder
        self.low_decoder = torch.nn.ModuleDict({
            'deconv2d_1': ConvTranspose2dCell(in_channels=128, out_channels=64, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_2': ConvTranspose2dCell(in_channels=64, out_channels=32, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_3': ConvTranspose2dCell(in_channels=32, out_channels=16, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_4': ConvTranspose2dCell(in_channels=16, out_channels=1, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=None, dropout=self.dropout_rate),
        })

        self.high_decoder = torch.nn.ModuleDict({
            'deconv2d_1': ConvTranspose2dCell(in_channels=256, out_channels=64, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_2': ConvTranspose2dCell(in_channels=128, out_channels=32, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_3': ConvTranspose2dCell(in_channels=64, out_channels=16, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_4': ConvTranspose2dCell(in_channels=32, out_channels=1, kernel_size=(5, 3), stride=(1, 2), padding=0, activation=None, dropout=self.dropout_rate),
        })

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Linear(258, 257)
        )

    def enhance_vc2(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        input = torch.nn.functional.pad(input, (0, 0, 16, 0))
        input = torch.log10(input + 1e-10)

        # input normalization
        input_feat = torch.permute(input, (0, 2, 1))
        input_feat = self.input_norm(input_feat)
        input_feat = torch.permute(input_feat, (0, 2, 1))

        B, K, D = input_feat.shape
        input_feat = torch.unsqueeze(input_feat, dim=1)
        hb_feats = torch.unsqueeze(self.harmonic_bank, dim=0).repeat(B, 1, 1)
        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        # Chunk along feats
        input_low, input_high = torch.split(input_feat, [129, 128], dim=-1)
        input_high = torch.nn.functional.pad(input_high, (0, 1, 0, 0))

        encoder_temp_low = [0]*4
        i = 0 
        for k, conv in self.low_encoder.items():
            if i == 0:
                encoder_temp_low[i] = conv(input_low)
            else:
                encoder_temp_low[i] = conv(encoder_temp_low[i-1])
            i += 1
        encode_low_feats = encoder_temp_low[i-1]

        # Middle Layers
        # reshape: flatten along with channels
        B, C, K, D = encode_low_feats.shape
        encode_low_feats = torch.permute(encode_low_feats, (0, 2, 1, 3))
        encode_low_feats = torch.reshape(encode_low_feats, (B, K, C*D))
        
        att_feats = encode_low_feats
        residual = encode_low_feats
        for i in range(len(self.attention)):
            att_feats = self.attention[i](att_feats, hb_feats, hb_feats, None)
            att_feats = 0.5 * (att_feats + residual)
            residual = att_feats
        
        # reshape
        decoder_low_input = torch.reshape(att_feats, (B, K, C, D))
        decoder_low_input = torch.permute(decoder_low_input, (0, 2, 1, 3))   
        
        i = 3
        for k, deconv in self.low_decoder.items():
            if decoder_low_input.shape[-1] < encoder_temp_low[i].shape[-1]:
                decoder_low_input = torch.nn.functional.pad(
                    decoder_low_input,
                    (0, encoder_temp_low[i].shape[-1] - decoder_low_input.shape[-1], 0, 0)
                )
            # print(decoder_input.shape[-1], encoder_temp[i].shape[-1])
            # decoder_low_input = torch.concat((decoder_low_input, encoder_temp_low[i]), dim=1)
            decoder_low_input = deconv(decoder_low_input)
            i -= 1

        encoder_temp_high = [0]*4
        i = 0 
        for k, conv in self.high_encoder.items():
            if i == 0:
                encoder_temp_high[i] = conv(input_high)
            else:
                encoder_temp_high[i] = conv(encoder_temp_high[i-1])
            i += 1
        encode_high_feats = encoder_temp_high[i-1]

        B, C, K, D = encode_high_feats.shape
        encode_high_feats = torch.permute(encode_high_feats, (0, 2, 1, 3))
        encode_high_feats = torch.reshape(encode_high_feats, (B, K, C*D))

        # h0 = torch.zeros(1, B, self.hidden_size, device=device)
        encode_high_feats = self.linear_high_in(encode_high_feats)
        rnn_high_feats = self.high_rnn(encode_high_feats)
        rnn_high_feats = self.linear_high_out(rnn_high_feats)

        decoder_high_input = torch.reshape(rnn_high_feats, (B, K, C, D))
        decoder_high_input = torch.permute(decoder_high_input, (0, 2, 1, 3))   
        
        i = 3
        for k, deconv in self.high_decoder.items():
            if decoder_high_input.shape[-1] < encoder_temp_high[i].shape[-1]:
                decoder_high_input = torch.nn.functional.pad(
                    decoder_high_input,
                    (0, encoder_temp_high[i].shape[-1] - decoder_high_input.shape[-1], 0, 0)
                )
            # print(decoder_input.shape[-1], encoder_temp[i].shape[-1])
            decoder_high_input = torch.concat((decoder_high_input, encoder_temp_high[i]), dim=1)
            decoder_high_input = deconv(decoder_high_input)
            i -= 1
        
        enhanced_feats = torch.concat([decoder_low_input, decoder_high_input], dim=-1)
        # enhanced_feats = enhanced_feats[:, :, :, :-1]
        enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = torch.squeeze(enhanced_feats, dim=1)
        enhanced_feats = enhanced_feats + input
        # enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = enhanced_feats[:, 16:, :]
        enhanced_feats = 10 ** enhanced_feats
        # enhanced_feats = enhanced_feats ** 2 # magnitude to power

        return enhanced_feats, None, None

    def build_ehance_model_vc3(self):
        self.d_feats = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,
        self.input_norm = nn.BatchNorm1d(self.d_feats)

        # Convolution 2d encoder
        self.high_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        self.low_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        # RNN 
        self.hidden_size = 512

        high_rnn_size = 256-124+1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = high_rnn_size * 128

        self.high_linear_in = nn.Linear(high_rnn_size, self.hidden_size)
        high_att_list = []
        for i in range(self.hb_att_layer):
            if self.hb_att_norm:
                high_att_list.append(nn.LayerNorm(self.hidden_size))
            high_att_list.append(
                GumbelMultiHeadedAttention(
                    1, self.hidden_size, self.d_feats // 2 + 1, self.K, self.hidden_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
        self.high_attention = nn.ModuleList(high_att_list)
        self.high_linear_out = nn.Linear(self.hidden_size, 960)

        low_rnn_size = self.d_feats // 2 + 1 + 4
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = low_rnn_size * 128
        
        self.low_linear_in = nn.Linear(low_rnn_size, self.hidden_size)
        low_att_list = []
        for i in range(self.hb_att_layer):
            if self.hb_att_norm:
                low_att_list.append(nn.LayerNorm(self.hidden_size))
            low_att_list.append(
                GumbelMultiHeadedAttention(
                    1, self.hidden_size, self.d_feats // 2 + 1, self.K, self.hidden_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
        self.low_attention = nn.ModuleList(low_att_list)
        
        self.low_linear_out = nn.Linear(self.hidden_size, 960)

        # De-convolution 2d decoder
        self.decoder = torch.nn.ModuleDict({
            # 'deconv2d_1': GatedConvTranspose2dCell(in_channels=256, out_channels=128, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_1': GatedConvTranspose2dCell(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_2': GatedConvTranspose2dCell(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_3': GatedConvTranspose2dCell(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_4': GatedConvTranspose2dCell(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate, norm=False),
        })

        self.output_layer = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Linear(257, 257),
        )

    def enhance_vc3(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        # input = torch.sqrt(input)
        input = torch.nn.functional.pad(input, (0, 0, 8, 0))
        input = torch.log10(input + 1e-10)

        # input normalization
        input = torch.permute(input, (0, 2, 1))
        input = self.input_norm(input)
        input = torch.permute(input, (0, 2, 1))

        B, K, D = input.shape
        input = torch.unsqueeze(input, dim=1)
        hb_feats = torch.unsqueeze(self.harmonic_bank, dim=0).repeat(B, 1, 1)
        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        # Chunk along feats
        input_low = input[:, :, :, :133]
        input_high = input[:, :, :, 124:]

        # low encoder
        encoder_temp_low = [0]*len(self.low_encoder)
        i = 0 
        for k, conv in self.low_encoder.items():
            if i == 0:
                encoder_temp_low[i] = conv(input_low)
            else:
                encoder_temp_low[i] = conv(encoder_temp_low[i-1])
            i += 1
        encode_low_feats = encoder_temp_low[i-1]

        # Middle Layers
        # reshape: flatten along with channels
        B, C, K, D = encode_low_feats.shape
        encode_low_feats = torch.permute(encode_low_feats, (0, 2, 1, 3))
        encode_low_feats = torch.reshape(encode_low_feats, (B, K, C*D))
        
        encode_low_feats = self.low_linear_in(encode_low_feats)
        low_att_feats = encode_low_feats
        residual = encode_low_feats

        low_att_feats_list = []
        for i in range(len(self.low_attention)):
            if self.hb_att_norm:
                if isinstance(self.low_attention[i], MultiHeadedAttention):
                    low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
                    low_att_feats = low_att_feats + residual
                    residual = low_att_feats
                elif isinstance(self.low_attention[i], nn.LayerNorm): # layernorm
                    low_att_feats = self.low_attention[i](low_att_feats)
            else:
                low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
                low_att_feats = low_att_feats + residual
                residual = low_att_feats
        low_att_feats = self.low_linear_out(low_att_feats)

        # High encoder
        encoder_temp_high = [0]*len(self.high_encoder)
        i = 0 
        for k, conv in self.high_encoder.items():
            if i == 0:
                encoder_temp_high[i] = conv(input_high)
            else:
                encoder_temp_high[i] = conv(encoder_temp_high[i-1])
            i += 1
        encode_high_feats = encoder_temp_high[i-1]

        B, C, K, D = encode_high_feats.shape
        encode_high_feats = torch.permute(encode_high_feats, (0, 2, 1, 3))
        encode_high_feats = torch.reshape(encode_high_feats, (B, K, C*D))

        encode_high_feats = self.high_linear_in(encode_high_feats)
        high_att_feats = encode_high_feats
        residual = encode_high_feats

        high_att_feats_list = []
        for i in range(len(self.high_attention)):
            if self.hb_att_norm:
                if isinstance(self.high_attention[i], MultiHeadedAttention):
                    high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
                    high_att_feats = high_att_feats + residual
                    residual = high_att_feats
                elif isinstance(self.high_attention[i], nn.LayerNorm): # layernorm
                    high_att_feats = self.high_attention[i](high_att_feats + residual)
            else:
                high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
                high_att_feats = high_att_feats + residual
                residual = high_att_feats
        high_att_feats = self.high_linear_out(high_att_feats)

        # # decoder_high_input = torch.reshape(rnn_high_feats, (B, K, C, D))
        decoder_input = torch.concat([low_att_feats, high_att_feats], dim=-1)
        # decoder_input = self.linear(att_feats)
        decoder_input = torch.reshape(decoder_input, (B, K, C, D*2+1))
        decoder_input = torch.permute(decoder_input, (0, 2, 1, 3))   
        
        i = len(self.decoder) - 1
        for k, deconv in self.decoder.items():
            if decoder_input.shape[-1] == 127:
                decoder_input = torch.nn.functional.pad(
                    decoder_input,
                    (0, 1, 0, 0)
                )
            # print(decoder_input.shape[-1], encoder_temp[i].shape[-1])
            # decoder_high_input = torch.concat((decoder_high_input, encoder_temp_high[i]), dim=1)
            decoder_input = deconv(decoder_input)
            i -= 1
        
        enhanced_feats = torch.squeeze(decoder_input, dim=1)
        enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = enhanced_feats[:, 8:, :]
        enhanced_feats = 10 ** enhanced_feats

        return enhanced_feats, None, None


    def build_ehance_model_vc4(self):
        self.d_feats = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,
        self.input_norm = nn.BatchNorm1d(self.d_feats)

        # Convolution 2d encoder
        self.high_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        self.low_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        # RNN 
        self.hidden_size = 512

        high_rnn_size = 256-124+1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = high_rnn_size * 128

        self.high_linear_in = nn.Linear(high_rnn_size, self.hidden_size)
        high_att_list = []
        for i in range(self.hb_att_layer):
            if self.hb_att_norm:
                high_att_list.append(nn.LayerNorm(self.hidden_size))
            high_att_list.append(
                GumbelMultiHeadedAttention(
                    1, self.hidden_size, self.d_feats // 2 + 1, self.K, self.hidden_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
        self.high_attention = nn.ModuleList(high_att_list)
        self.high_linear_out = nn.Linear(self.hidden_size, 960)

        low_rnn_size = self.d_feats // 2 + 1 + 4
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = low_rnn_size * 128
        
        self.low_linear_in = nn.Linear(low_rnn_size, self.hidden_size)
        low_att_list = []
        for i in range(self.hb_att_layer):
            if self.hb_att_norm:
                low_att_list.append(nn.LayerNorm(self.hidden_size))
            low_att_list.append(
                GumbelMultiHeadedAttention(
                    1, self.hidden_size, self.d_feats // 2 + 1, self.K, self.hidden_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
        self.low_attention = nn.ModuleList(low_att_list)
        
        self.low_linear_out = nn.Linear(self.hidden_size, 960)

        # De-convolution 2d decoder
        self.decoder = torch.nn.ModuleDict({
            # 'deconv2d_1': GatedConvTranspose2dCell(in_channels=256, out_channels=128, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_1': GatedConvTranspose2dCell(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_2': GatedConvTranspose2dCell(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_3': GatedConvTranspose2dCell(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_4': GatedConvTranspose2dCell(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate, norm=False),
        })

        self.output_layer = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Linear(257, 257),
        )

    def enhance_vc4(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        # input = torch.sqrt(input)
        input = torch.nn.functional.pad(input, (0, 0, 8, 0))
        input = torch.log10(input + 1e-10)

        # input normalization
        input = torch.permute(input, (0, 2, 1))
        input = self.input_norm(input)
        input = torch.permute(input, (0, 2, 1))

        B, K, D = input.shape
        input = torch.unsqueeze(input, dim=1)
        hb_feats = torch.unsqueeze(self.harmonic_bank, dim=0).repeat(B, 1, 1)
        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        # Chunk along feats
        input_low = input[:, :, :, :133]
        input_high = input[:, :, :, 124:]
        # input_low, input_high = torch.split(input, [129, 128], dim=-1)
        # input_high = torch.nn.functional.pad(input_high, (0, 1, 0, 0))

        # low encoder
        encoder_temp_low = [0]*len(self.low_encoder)
        i = 0 
        for k, conv in self.low_encoder.items():
            if i == 0:
                encoder_temp_low[i] = conv(input_low)
            else:
                encoder_temp_low[i] = conv(encoder_temp_low[i-1])
            i += 1
        encode_low_feats = encoder_temp_low[i-1]

        # Middle Layers
        # reshape: flatten along with channels
        B, C, K, D = encode_low_feats.shape
        encode_low_feats = torch.permute(encode_low_feats, (0, 2, 1, 3))
        encode_low_feats = torch.reshape(encode_low_feats, (B, K, C*D))
        
        encode_low_feats = self.low_linear_in(encode_low_feats)
        low_att_feats = encode_low_feats
        residual = encode_low_feats
        # for i in range(len(self.low_attention)):
            # if self.hb_att_norm:
            #     if isinstance(self.low_attention[i], MultiHeadedAttention):
            #         low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
            #     elif isinstance(self.low_attention[i], nn.LayerNorm): # layernorm
            #         low_att_feats = self.low_attention[i](low_att_feats + residual)
            #         residual = low_att_feats
            # else:
            #     low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
            #     low_att_feats = low_att_feats + residual
            #     residual = low_att_feats
        low_att_feats_list = []
        for i in range(len(self.low_attention)):
            if self.hb_att_norm:
                if isinstance(self.low_attention[i], MultiHeadedAttention):
                    low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
                    low_att_feats_list.append(low_att_feats)
                    if i != len(self.low_attention) - 1:
                        low_att_feats = low_att_feats + residual
                        residual = low_att_feats
                elif isinstance(self.low_attention[i], nn.LayerNorm): # layernorm
                    low_att_feats = self.low_attention[i](low_att_feats)
            else:
                low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
                low_att_feats = low_att_feats + residual
                residual = low_att_feats
        # att_feats = self.attention[-1](att_feats, hb_feats, hb_feats, None)
        low_att_feats = torch.stack(low_att_feats_list, dim=0).sum(dim=0)
        low_att_feats = self.low_linear_out(low_att_feats)

        # High encoder
        encoder_temp_high = [0]*len(self.high_encoder)
        i = 0 
        for k, conv in self.high_encoder.items():
            if i == 0:
                encoder_temp_high[i] = conv(input_high)
            else:
                encoder_temp_high[i] = conv(encoder_temp_high[i-1])
            i += 1
        encode_high_feats = encoder_temp_high[i-1]

        B, C, K, D = encode_high_feats.shape
        encode_high_feats = torch.permute(encode_high_feats, (0, 2, 1, 3))
        encode_high_feats = torch.reshape(encode_high_feats, (B, K, C*D))

        encode_high_feats = self.high_linear_in(encode_high_feats)
        high_att_feats = encode_high_feats
        residual = encode_high_feats
        # for i in range(len(self.high_attention)):
        #     if self.hb_att_norm:
        #         if isinstance(self.high_attention[i], MultiHeadedAttention):
        #             high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
        #         elif isinstance(self.high_attention[i], nn.LayerNorm): # layernorm
        #             high_att_feats = self.high_attention[i](high_att_feats + residual)
        #             residual = high_att_feats
        #     else:
        #         high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
        #         high_att_feats = high_att_feats + residual
        #         residual = high_att_feats
        high_att_feats_list = []
        for i in range(len(self.high_attention)):
            if self.hb_att_norm:
                if isinstance(self.high_attention[i], MultiHeadedAttention):
                    high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
                    high_att_feats_list.append(high_att_feats)
                    if i != len(self.high_attention) - 1:
                        high_att_feats = high_att_feats + residual
                        residual = high_att_feats
                elif isinstance(self.high_attention[i], nn.LayerNorm): # layernorm
                    high_att_feats = self.high_attention[i](high_att_feats + residual)
            else:
                high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
                high_att_feats = high_att_feats + residual
                residual = high_att_feats
        high_att_feats = torch.stack(high_att_feats_list, dim=0).sum(dim=0)
        high_att_feats = self.high_linear_out(high_att_feats)

        # # decoder_high_input = torch.reshape(rnn_high_feats, (B, K, C, D))
        decoder_input = torch.concat([low_att_feats, high_att_feats], dim=-1)
        # decoder_input = self.linear(att_feats)
        decoder_input = torch.reshape(decoder_input, (B, K, C, D*2+1))
        decoder_input = torch.permute(decoder_input, (0, 2, 1, 3))   
        
        i = len(self.decoder) - 1
        for k, deconv in self.decoder.items():
            if decoder_input.shape[-1] == 127:
                decoder_input = torch.nn.functional.pad(
                    decoder_input,
                    (0, 1, 0, 0)
                )
            # print(decoder_input.shape[-1], encoder_temp[i].shape[-1])
            # decoder_high_input = torch.concat((decoder_high_input, encoder_temp_high[i]), dim=1)
            decoder_input = deconv(decoder_input)
            i -= 1
        
        # enhanced_feats = torch.concat([decoder_low_input, decoder_high_input], dim=-1)
        # enhanced_feats = enhanced_feats[:, :, :, :-1]
        # enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = torch.squeeze(decoder_input, dim=1)
        enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = enhanced_feats[:, 8:, :]
        enhanced_feats = 10 ** enhanced_feats
        # enhanced_feats = enhanced_feats ** 2 # magnitude to power

        return enhanced_feats, None, None

    def generate_harmonic_bank(self, half=False):
        n_fft = self.n_fft
        fs = self.fs
        # up_freq = self.up_freq
        if half == True:
            d_feats = n_fft // 4 + 1
        else:
            d_feats = n_fft // 2 + 1
        
        freq_max = 4000
        times = 100
        length = 16000

        # Convert freq to tone scale
        freq_start = self.hb_start
        freq_end = self.hb_end
        tone_scale = 24
        x_start = math.ceil(tone_scale * math.log2(freq_start))
        x_end = math.floor(tone_scale * math.log2(freq_end))
        freq_bins = [freq for freq in range(x_start, x_end+1, 1)]
        K = len(freq_bins) + 1

        harmonic_bank = np.zeros([K, d_feats], dtype=np.float32)
        sample_pos = int((16000 / self.hop_length + 1 ) / 2)
        count = 1
        for bin in freq_bins:
            # Reverse to freq
            i = round(math.pow(2, bin/tone_scale))
            tmp_freq_max = freq_max if i * times > freq_max else i * times
            tmp_freq = [j for j in range(i, tmp_freq_max+1, i)]
            # num_freq = len(tmp_freq)
            sine = np.zeros(16000)
            for freq in tmp_freq:
                sine += librosa.tone(freq, sr=16000, length=length)
            sine = self.norm(sine)
            # sine = sine / num_freq
            S = librosa.stft(y=sine, 
                             n_fft=self.n_fft, 
                             win_length=self.n_fft, 
                             hop_length=self.hop_length, 
                             window=self.window,
                             pad_mode='reflect')
            power = S.real**2 + S.imag ** 2
            power = power.transpose()
            # power /= num_freq
            # print(power)
            harmonic_bank[count, :] = power[sample_pos, :d_feats]
            # power[sample_pos, :] = np.log10(power[sample_pos, :] + 0.1)
            # power[sample_pos, :] = power[sample_pos, :] / np.sum(np.absolute(power[sample_pos, :]))
            # harmonic_bank[count, :] = power[sample_pos, :]
            count += 1
        self.K = K
        harmonic_bank = torch.from_numpy(harmonic_bank)
        # harmonic_bank = torch.sqrt(harmonic_bank)
        harmonic_bank = torch.log10(harmonic_bank + 1e-10)
        self.register_buffer("harmonic_bank", harmonic_bank)

    def norm(self, data):
        max = data.max()
        min = data.min()
        dis = max - min
        data = data - min
        data = data / dis
        data = data * 2 - 1
        return data

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        input_power_clean = None

        if self.return_spec:
            input_stft_clean, _ = self._compute_stft(input, input_lengths)
            if self.irm_source:
                input_magn_clean = torch.sqrt(input_stft_clean.real ** 2 + input_stft_clean.imag ** 2)
                with torch.no_grad(): # Get irm source to calculate loss
                    input_magn_clean = self.irm(input_magn_clean)
                    input_power_clean = input_magn_clean ** 2
            else:
                input_power_clean = input_stft_clean.real ** 2 + input_stft_clean.imag ** 2

        if self.add_awgn and self.training:
            # alpha = input.abs().max(dim=-1, keepdim=True).values * 0.1
            # awgn1 = alpha * torch.randn(input.shape, device=device)
            # awgn2 = alpha * torch.randn(input.shape, device=device)
            # awgn3 = alpha * torch.randn(input.shape, device=device)
            input = add_awgn_batch(input, self.snr)
            
        input_stft, feats_lens = self._compute_stft(input, input_lengths)
        input_power = input_stft.real ** 2 + input_stft.imag ** 2
        enhanced_power, _, _ = self.enhance(input_power)
        enhanced_wav, enhanced_wav_lens = self._to_wavform(enhanced_power, input_stft, input_lengths)

        if self.irm_source:
            feats_intermediate = (input_power_clean[:, :, 129:], enhanced_power[:, :, 129:])
        else:
            feats_intermediate = (input_power_clean, enhanced_power)

        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(enhanced_wav, enhanced_wav_lens)
        else:
            input_stft = ComplexTensor(enhanced_wav[..., 0], enhanced_wav[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        if self.return_spec:
            return input_feats, feats_lens, feats_intermediate
        else:
            return input_feats, feats_lens
        
        # # padding
        # current_time = enhanced_wav.size(1)
        # source_time = input.size(1)
        # padding_length = source_time - current_time  # How much padding is needed
        # if padding_length > 0:
        #     # Apply padding along the time dimension
        #     # F.pad takes (left_pad, right_pad) for each dimension in reverse order
        #     enhanced_wav = nn.functional.pad(enhanced_wav, (0, padding_length))  # Pad at the end of the time dimension
        enhanced_stft, enhanced_len = self._compute_stft(enhanced_wav, enhanced_wav_lens)
        enhanced_power = enhanced_stft.real**2 + enhanced_stft.imag**2
        
    def _to_wavform(self, input_power, input_stft, input_lens):
        input_phase = torch.complex(input_stft.real, input_stft.imag)
        input_phase = torch.angle(input_phase)

        r = torch.sqrt(input_power) # convert power spectrum to magnitude spectrum
        cos = torch.cos(input_phase)
        sin = torch.sin(input_phase)
        norm = torch.complex(cos, sin)
        
        spec_with_phase = torch.mul(r, norm)
        wav, wav_lens = self.stft.inverse(spec_with_phase, input_lens)
        # wav = self._unitize(wav)
        wav = nn.functional.tanh(wav)
        # if not self.training:
        #     wav = nn.functional.softshrink(wav, lambd=0.2)

        return wav, wav_lens

    def _unitize(self, input: torch.Tensor):       
        max = input.abs().max(dim=1, keepdim=True).values
        min = -max
        # min = input.min(dim=1, keepdim=True).values
        # max = input.max(dim=1, keepdim=True).values

        range_values = max - min
        range_values[range_values == 0] = 1e-8  # Replace zeros with a small value

        unitized_data = ((input - min) / range_values) * 2 -1

        return unitized_data
    
    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens 
    
    def output_size(self) -> int:
        return self.n_mels

class ComplexFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Log-Mel-Fbank
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,# Create a mask using tensor operations
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        type = 'cvc_v1',
        dropout_rate = 0.0,
        use_gumbel=True,
        add_awgn=False,
        snr=20,
        hb_start=100,
        hb_end=300,
        hb_att_layer=2,
        hb_att_norm=False,
        irm_source=False,
    ):
        # super().__init__(fs, n_fft, win_length, hop_length, window, center, normalized,
        #                  onesided, n_mels, fmin, fmax, htk, frontend_conf, apply_stft)
        super().__init__()

        self.return_spec = False
        
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length

        self.fs = fs
        self.n_fft = n_fft
        self.window = window
        self.dropout_rate = dropout_rate
        self.use_gumbel = use_gumbel
        self.irm_source = irm_source

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.add_awgn = add_awgn
        self.snr = snr
        self.hb_start = hb_start
        self.hb_end = hb_end
        self.hb_att_layer = hb_att_layer
        self.hb_att_norm = hb_att_norm

        # self.irm_source = irm_source
        # if irm_source:
        #     self.irm = IRMBlock(n_fft=n_fft, dropout_rate=0.0)

        if type == 'cvc_v1':
            self.generate_harmonic_bank(half=True)
            self.build_ehance_model_cvc1()
            self.enhance = self.enhance_cvc1

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

    def build_ehance_model_cvc1(self):
        self.real_can_block = CANBlock(
            self.n_fft,
            self.dropout_rate,
            self.K,
            self.use_gumbel,
            self.hb_att_layer,
            self.hb_att_norm,
            self.irm_source
        )
        self.imag_can_block = CANBlock(
            self.n_fft,
            self.dropout_rate,
            self.K,
            self.use_gumbel,
            self.hb_att_layer,
            self.hb_att_norm,
            self.irm_source,
        )

    def enhance_cvc1(self, input_real, input_imag):
        enhanced_real = self.real_can_block(input_real, self.harmonic_bank)
        enhanced_imag = self.imag_can_block(input_imag, self.harmonic_bank)

        return ComplexTensor(enhanced_real, enhanced_imag)

    def generate_harmonic_bank(self, half=False):
        n_fft = self.n_fft
        fs = self.fs
        # up_freq = self.up_freq
        if half == True:
            d_feats = n_fft // 4 + 1
        else:
            d_feats = n_fft // 2 + 1
        
        freq_max = 4000
        times = 100
        length = 16000

        # Convert freq to tone scale
        freq_start = self.hb_start
        freq_end = self.hb_end
        tone_scale = 24
        x_start = math.ceil(tone_scale * math.log2(freq_start))
        x_end = math.floor(tone_scale * math.log2(freq_end))
        freq_bins = [freq for freq in range(x_start, x_end+1, 1)]
        K = len(freq_bins) + 1

        harmonic_bank = np.zeros([K, d_feats], dtype=np.float32)
        sample_pos = int((16000 / self.hop_length + 1 ) / 2)
        count = 1
        for bin in freq_bins:
            # Reverse to freq
            i = round(math.pow(2, bin/tone_scale))
            tmp_freq_max = freq_max if i * times > freq_max else i * times
            tmp_freq = [j for j in range(i, tmp_freq_max+1, i)]
            # num_freq = len(tmp_freq)
            sine = np.zeros(16000)
            for freq in tmp_freq:
                sine += librosa.tone(freq, sr=16000, length=length)
            sine = self.norm(sine)
            # sine = sine / num_freq
            S = librosa.stft(y=sine, 
                             n_fft=self.n_fft, 
                             win_length=self.n_fft, 
                             hop_length=self.hop_length, 
                             window=self.window,
                             pad_mode='reflect')
            power = S.real**2 + S.imag ** 2
            power = power.transpose()
            # power /= num_freq
            # print(power)
            harmonic_bank[count, :] = power[sample_pos, :d_feats]
            # power[sample_pos, :] = np.log10(power[sample_pos, :] + 0.1)
            # power[sample_pos, :] = power[sample_pos, :] / np.sum(np.absolute(power[sample_pos, :]))
            # harmonic_bank[count, :] = power[sample_pos, :]
            count += 1
        self.K = K
        harmonic_bank = torch.from_numpy(harmonic_bank)
        # harmonic_bank = torch.sqrt(harmonic_bank)
        harmonic_bank = torch.log10(harmonic_bank + 1e-10)
        self.register_buffer("harmonic_bank", harmonic_bank)

    def norm(self, data):
        max = data.max()
        min = data.min()
        dis = max - min
        data = data - min
        data = data / dis
        data = data * 2 - 1
        return data

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        input_power_clean = None

        if self.return_spec:
            input_stft_clean, _ = self._compute_stft(input, input_lengths)
            if self.irm_source:
                input_magn_clean = torch.sqrt(input_stft_clean.real ** 2 + input_stft_clean.imag ** 2)
                with torch.no_grad(): # Get irm source to calculate loss
                    input_magn_clean = self.irm(input_magn_clean)
                    input_power_clean = input_magn_clean ** 2
            else:
                input_power_clean = input_stft_clean.real ** 2 + input_stft_clean.imag ** 2

        if self.add_awgn and self.training:
            # alpha = input.abs().max(dim=-1, keepdim=True).values * 0.1
            # awgn1 = alpha * torch.randn(input.shape, device=device)
            # awgn2 = alpha * torch.randn(input.shape, device=device)
            # awgn3 = alpha * torch.randn(input.shape, device=device)
            input = add_awgn_batch(input, self.snr)
            
        input_stft, feats_lens = self._compute_stft(input, input_lengths)
        enhanced_stft = self.enhance(input_stft.real, input_stft.imag)
        # input_power = input_stft.real ** 2 + input_stft.imag ** 2
        enhanced_wav, enhanced_wav_lens = self._to_wavform(enhanced_stft, input_lengths)
        enhanced_power = enhanced_stft.real ** 2 + enhanced_stft.imag ** 2

        if self.irm_source:
            feats_intermediate = (input_power_clean[:, :, 129:], enhanced_power[:, :, 129:])
        else:
            feats_intermediate = (input_power_clean, enhanced_power)

        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(enhanced_wav, enhanced_wav_lens)
        else:
            input_stft = ComplexTensor(enhanced_wav[..., 0], enhanced_wav[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        if self.return_spec:
            return input_feats, feats_lens, feats_intermediate
        else:
            return input_feats, feats_lens
        
        # # padding
        # current_time = enhanced_wav.size(1)
        # source_time = input.size(1)
        # padding_length = source_time - current_time  # How much padding is needed
        # if padding_length > 0:
        #     # Apply padding along the time dimension
        #     # F.pad takes (left_pad, right_pad) for each dimension in reverse order
        #     enhanced_wav = nn.functional.pad(enhanced_wav, (0, padding_length))  # Pad at the end of the time dimension
        enhanced_stft, enhanced_len = self._compute_stft(enhanced_wav, enhanced_wav_lens)
        enhanced_power = enhanced_stft.real**2 + enhanced_stft.imag**2
        
    def _to_wavform(self, input_stft, input_lens):
        wav, wav_lens = self.stft.inverse(input_stft, input_lens)
        # wav = self._unitize(wav)
        wav = nn.functional.tanh(wav)
        # if not self.training:
        #     wav = nn.functional.softshrink(wav, lambd=0.2)

        return wav, wav_lens

    def _unitize(self, input: torch.Tensor):       
        max = input.abs().max(dim=1, keepdim=True).values
        min = -max
        # min = input.min(dim=1, keepdim=True).values
        # max = input.max(dim=1, keepdim=True).values

        range_values = max - min
        range_values[range_values == 0] = 1e-8  # Replace zeros with a small value

        unitized_data = ((input - min) / range_values) * 2 -1

        return unitized_data
    
    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens 
    
    def output_size(self) -> int:
        return self.n_mels
    
def add_awgn_batch(audio_tensor, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to a batch of audio signals.

    Parameters:
        audio_tensor (torch.Tensor): Batch of audio signals of shape (batch_size, length).
        snr_db (float): Desired Signal-to-Noise Ratio in decibels.

    Returns:
        torch.Tensor: Batch of audio signals with AWGN added.
    """
    # Compute signal power for each audio in the batch (along the length dimension)
    signal_power = torch.mean(audio_tensor**2, dim=1, keepdim=True)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Compute noise power for each signal
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise (same shape as the audio batch)
    noise = torch.randn_like(audio_tensor) * torch.sqrt(noise_power)

    # Add noise to the audio signals in the batch
    noisy_audio = audio_tensor + noise

    return noisy_audio

class Conv2dCell(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, padding = 0, dropout=0.0):
        super().__init__()
        self.cnn = torch.nn.Conv2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
        self.normalize  = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.normalize(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class GatedConv2dCell(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0, dropout=0.0):
        super().__init__()
        self.cnn = torch.nn.Conv2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
        
        self.normalize  = torch.nn.BatchNorm2d(out_channels)

        self.gate = torch.nn.Conv2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
        self.gate_bn = nn.BatchNorm2d(out_channels)
        self.gate_out = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # CNN
        y = self.cnn(x)
        y = self.normalize(y)
        # Gate
        g = self.gate(x)
        g = self.gate_bn(g)
        g = self.gate_out(g)

        y = y * g
        y = self.dropout(y)
        return y

class ConvTranspose2dCell(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, groups=1, padding = 0, dropout=0.0):
        super().__init__()
        self.cnn = torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            groups=groups)
        if activation != None:
            self.normalize  = torch.nn.BatchNorm2d(out_channels)

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.cnn(x)
        if self.activation != None:
            x = self.normalize(x)
            x = self.activation(x)
        x = self.dropout(x)
        return x
    
class GatedConvTranspose2dCell(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, padding = 0, dropout=0.0, norm=True):
        super().__init__()
        self.cnn = torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            groups=groups)
        
        self.norm = norm
        if norm:
            self.normalize  = torch.nn.BatchNorm2d(out_channels)

        self.gate = torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             groups=groups)
        self.gate_bn = nn.BatchNorm2d(out_channels)
        self.gate_out = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # CNN
        y = self.cnn(x)
        if self.norm:
            y = self.normalize(y)
        # Gate
        g = self.gate(x)
        g = self.gate_bn(g)
        g = self.gate_out(g)

        y = y * g
        y = self.dropout(y)
        return y

class ConvTranspose2dDecoderCell(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, padding = 0, dropout=0.0):
        super().__init__()
        self.cnn1 = torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding)
        self.cnn2 = torch.nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding)
        # self.normalize1  = torch.nn.BatchNorm2d(out_channels)
        self.normalize  = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x1, x2):
        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        x = x1 + x2
        x = self.normalize(x)
        
        # x1 = self.normalize1(x1)
        # x2 = self.normalize2(x2)
        if self.activation != None:
            x = self.activation(x)
        x = self.dropout(x)
        return x
    
class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.ModuleDict({
            'conv2d_1': Conv2dCell(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'conv2d_2': Conv2dCell(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'conv2d_3': Conv2dCell(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'conv2d_4': Conv2dCell(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'conv2d_5': Conv2dCell(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
        })

    def forward(self, x):
        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        encoder_temp = []
        for k, conv in self.encoder.items():
            x = conv(x)
            encoder_temp.append(x)
        return x, encoder_temp

class CNNDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # De-convolution 2d decoder
        self.decoder = torch.nn.ModuleDict({
            'deconv2d_1': ConvTranspose2dCell(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'deconv2d_2': ConvTranspose2dCell(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'deconv2d_3': ConvTranspose2dCell(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'deconv2d_4': ConvTranspose2dCell(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2), activation=nn.ELU()),
            'deconv2d_5': ConvTranspose2dCell(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2), activation=nn.ReLU()),
        })
    
    def forward(self, x, encoder_output):
        i = 4
        mask = x
        for k, deconv in self.decoder.items():
            if mask.shape[-1] != encoder_output[i].shape[-1]:
                mask = torch.nn.functional.pad(
                    mask,
                    (0, 1, 0, 0)
                )
            mask = torch.concat((mask, encoder_output[i]), dim=1)
            mask = deconv(mask)
            i -= 1
        return mask

class AttentionBlock(nn.Module):
    # Implement cross attention
    def __init__(
            self,
            input_size,
            hidden_size,
            reference_size,
            # reference_len,
            output_size,
        ):
        super().__init__()

        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.reference_size = reference_size
        self.output_size    = output_size

        self.query = nn.Linear(input_size, hidden_size)
        self.key   = nn.Linear(reference_size, hidden_size)
        self.value = nn.Linear(reference_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = nn.BatchNorm2d(1)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, output_size)
        )

        self.norm2 = nn.BatchNorm2d(1)
        # self.linear = nn.Linear(hidden_size, output_size)
        # self.output_layer = nn.Sigmoid()
    
    def forward(self, x, ref):
        # x = torch.unsqueeze(x, dim=1)
        # x = self.norm1(x)
        # x = torch.squeeze(x, dim=1)

        # Norm first
        x = torch.unsqueeze(x, dim=1)
        x = self.norm1(x)
        x = torch.squeeze(x, dim=1)
        queries = self.query(x)
        keys    = self.key(ref)
        values  = self.value(ref)

        y = torch.matmul(queries, torch.transpose(keys, -2, -1)) / math.sqrt(self.reference_size)
        y = self.softmax(y)

        res = torch.matmul(y, values)

        # print(queries)
        # m = 0.5 * (x + m)
        # m = torch.unsqueeze(m, dim=1)
        # m = self.norm1(m)
        # m = torch.squeeze(m, dim=1)
        y = self.feed_forward(res)
        # y = y + res
        y = torch.unsqueeze(y, dim=1)
        y = self.norm2(y)
        y = torch.squeeze(y, dim=1)
        # y = self.linear(y)
        # y = self.output_layer(y)

        return y

class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            dropout=0.1,
            num_heads=1,
        ) -> None:
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout     = dropout
        self.num_heads   = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        self.norm1 = nn.BatchNorm2d(1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm2 = nn.BatchNorm2d(1)
    
    def forward(self, x):
        y, _ = self.mha(x, x, x)
        y = x + y
        y = torch.unsqueeze(y, dim=1)
        y = self.norm1(y)
        y = torch.squeeze(y, dim=1)

        y_f = self.ffn(y)
        y_f = y + y_f
        y_f = torch.unsqueeze(y_f, dim=1)
        y_f = self.norm2(y_f)
        y_f = torch.squeeze(y_f, dim=1)

        return y_f

class GumbelMultiHeadedAttention(MultiHeadedAttention):

    def __init__(
        self, 
        n_head, 
        n_src,
        n_ref,
        n_feat, 
        output_size=None,
        dropout_rate=0.0,
        use_gumbel=True):
        """Construct an GumbelMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.output_size = output_size
        # if output_size == None:
        #     self.linear_out = nn.Linear(n_feat, n_feat)
        # elif output_size == 0:
        #     self.linear_out = None
        # else:
        #     self.linear_out = nn.Linear(n_feat, output_size)
        self.linear_out = None
        self.d_v = output_size // n_head

        self.linear_q = nn.Linear(n_src, n_feat)
        self.linear_k = nn.Sequential(
            nn.Linear(n_ref, 256),
            nn.ELU(),
            nn.Linear(256, n_feat)
        )
        self.linear_v = nn.Sequential(
            nn.Linear(n_ref, 256),
            nn.ELU(),
            nn.Linear(256, output_size)
        )

        self.use_gumbel = use_gumbel

    def forward_qkv(self, query, key, value, expand_kv=False):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)

        if expand_kv:
            k_shape = key.shape
            k = (
                self.linear_k(key[:1, :, :])
                .expand(n_batch, k_shape[1], k_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
            v_shape = value.shape
            v = (
                self.linear_v(value[:1, :, :])
                .expand(n_batch, v_shape[1], v_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
        else:
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_v)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            if self.use_gumbel:
                self.attn = torch.nn.functional.gumbel_softmax(
                    scores, 1.0, hard=True, dim=-1
                ).masked_fill(
                    mask, 0.0
                )  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                )  # (batch, head, time1, time2)
        else:
            if self.use_gumbel:
                self.attn = torch.nn.functional.gumbel_softmax(
                    scores, 1.0, hard=True, dim=-1
                )  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2) (batch, head, time1, time2)


        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_v)
        )  # (batch, time1, d_model)

        # if self.linear_out != None:
        #     x = self.linear_out(x)
        return  x # (batch, time1, d_model)

    def forward(self, query, key, value, mask, expand_kv=False):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.
        When set to `True`, `Linear` layers are computed only for the first batch.
        This is useful to reduce the memory usage during decoding when the batch size is
        #beam_size x #mask_count, which can be very large. Typically, in single waveform
        inference of PAR, `Linear` layers should not be computed for all batches
        for source-attention.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value, expand_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class GumbelMultiHeadedAttention2(MultiHeadedAttention):

    def __init__(
        self, 
        n_head, 
        n_src,
        n_ref,
        n_feat, 
        output_size=None,
        dropout_rate=0.0,
        use_gumbel=True):
        """Construct an GumbelMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.output_size = output_size
        # if output_size == None:
        #     self.linear_out = nn.Linear(n_feat, n_feat)
        # elif output_size == 0:
        #     self.linear_out = None
        # else:
        #     self.linear_out = nn.Linear(n_feat, output_size)
        self.linear_out = None
        self.d_v = output_size // n_head

        self.linear_q = nn.Linear(n_src, n_feat)
        self.linear_k = nn.Sequential(
            nn.Linear(n_ref, 256),
            nn.ELU(),
            nn.Linear(256, n_feat)
        )
        self.linear_v = None
        self.use_gumbel = use_gumbel

    def forward_qkv(self, query, key, value, expand_kv=False):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)

        if expand_kv:
            k_shape = key.shape
            k = (
                self.linear_k(key[:1, :, :])
                .expand(n_batch, k_shape[1], k_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
            v_shape = value.shape
            v = (
                self.linear_v(value[:1, :, :])
                .expand(n_batch, v_shape[1], v_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
        else:
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            v = value.view(n_batch, -1, self.h, self.d_v)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            if self.use_gumbel:
                self.attn = torch.nn.functional.gumbel_softmax(
                    scores, 1.0, hard=True, dim=-1
                ).masked_fill(
                    mask, 0.0
                )  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                )  # (batch, head, time1, time2)
        else:
            if self.use_gumbel:
                self.attn = torch.nn.functional.gumbel_softmax(
                    scores, 1.0, hard=True, dim=-1
                )  # (batch, head, time1, time2)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2) (batch, head, time1, time2)


        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_v)
        )  # (batch, time1, d_model)

        # if self.linear_out != None:
        #     x = self.linear_out(x)
        return  x # (batch, time1, d_model)

    def forward(self, query, key, value, mask, expand_kv=False):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.
        When set to `True`, `Linear` layers are computed only for the first batch.
        This is useful to reduce the memory usage during decoding when the batch size is
        #beam_size x #mask_count, which can be very large. Typically, in single waveform
        inference of PAR, `Linear` layers should not be computed for all batches
        for source-attention.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value, expand_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class IRMBlock(nn.Module):
    def __init__(self, n_fft, dropout_rate, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_fft = n_fft
        self.dropout_rate = dropout_rate

        input_dim = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,

        # Convolution 2d encoder
        self.encoder = torch.nn.ModuleDict({
            'conv2d_1': Conv2dCell(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_2': Conv2dCell(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_3': Conv2dCell(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_4': Conv2dCell(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'conv2d_5': Conv2dCell(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
        })
        # RNN 
        self.hidden_size = 128
        self.d_feats = self.n_fft // 2 + 1
        rnn_input_size = self.d_feats
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = math.floor((rnn_input_size - (3 - 1) - 1) / 2 + 1)
        rnn_input_size = rnn_input_size * 256
        self.rnn_input_size = rnn_input_size
        self.rnn = torch.nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout_rate)
        self.linear = nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.rnn_input_size),
            nn.Dropout(self.dropout_rate),
        )
    
        # De-convolution 2d decoder
        self.decoder = torch.nn.ModuleDict({
            'deconv2d_1': ConvTranspose2dCell(in_channels=512, out_channels=128, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_2': ConvTranspose2dCell(in_channels=256, out_channels=64, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_3': ConvTranspose2dCell(in_channels=128, out_channels=32, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_4': ConvTranspose2dCell(in_channels=64, out_channels=16, kernel_size=(2, 3), stride=(1, 2), activation=nn.ELU(), dropout=self.dropout_rate),
            'deconv2d_5': ConvTranspose2dCell(in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2), activation=nn.Softplus(), dropout=self.dropout_rate),
        })

    def forward(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        input = torch.nn.functional.pad(input, (0, 0, 5, 0))
        # input = torch.sqrt(input) # power to magnitude
        # input = torch.log10(input + 0.00001)
        input_feats = torch.unsqueeze(input, dim=1)

        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        encoder_temp = [0]*5
        i = 0
        for k, conv in self.encoder.items():
            if i == 0:
                encoder_temp[i] = conv(input_feats)
            else:
                encoder_temp[i] = conv(encoder_temp[i-1])
            i += 1
        encode_feats = encoder_temp[i-1]

        # reshape: flatten along with channels
        B, C, K, D = encode_feats.shape
        encode_feats = torch.permute(encode_feats, (0, 2, 1, 3))
        encode_feats = torch.reshape(encode_feats, (B, K, C*D))

        # RNN
        device = encode_feats.device
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            h0 = torch.zeros(2, B, self.hidden_size, device=device)
            # c0 = torch.zeros(2, B, self.hidden_size, device=device)
            rnn_feats, _ = self.rnn(encode_feats, h0)
            rnn_feats = self.linear(rnn_feats)

        # reshape
        decoder_input = torch.reshape(rnn_feats, (B, K, C, D))
        decoder_input = torch.permute(decoder_input, (0, 2, 1, 3))   
        
        i = 4
        for k, deconv in self.decoder.items():
            if decoder_input.shape[-1] != encoder_temp[i].shape[-1]:
                decoder_input = torch.nn.functional.pad(
                    decoder_input,
                    (0, 1, 0, 0)
                )
            decoder_input = torch.concat((decoder_input, encoder_temp[i]), dim=1)
            decoder_input = deconv(decoder_input)
            i -= 1
        enhanced_feats = torch.squeeze(decoder_input, dim=1)
        # enhanced_feats = enhanced_feats ** 2
        enhanced_feats = input * enhanced_feats
        # enhanced_feats = enhanced_feats ** 2
        enhanced_feats = enhanced_feats[:, 5:, :]

        return enhanced_feats

class CANBlock(nn.Module):
    def __init__(
            self,
            n_fft, 
            dropout_rate,
            K,
            use_gumbel=True,
            hb_att_layer=2,
            hb_att_norm=False,
            irm_source=False,
        ):
        super().__init__()
        self.n_fft = n_fft
        self.dropout_rate = dropout_rate
        self.use_gumbel = use_gumbel
        self.hb_att_layer = hb_att_layer
        self.hb_att_norm = hb_att_norm
        self.K = K
        self.irm_source = irm_source
    
        self.d_feats = self.n_fft // 2 + 1
        
        # (batchs, channels, frames, frequency)
        # summary freqency information
        # along with freqency,
        self.input_norm = nn.BatchNorm1d(self.d_feats)

        # Convolution 2d encoder
        self.high_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        self.low_encoder = torch.nn.ModuleDict({
            'conv2d_1': GatedConv2dCell(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_2': GatedConv2dCell(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_3': GatedConv2dCell(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'conv2d_4': GatedConv2dCell(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
        })

        # RNN 
        self.hidden_size = 512

        high_rnn_size = 256-124+1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # high_rnn_size = (high_rnn_size - (3 - 1) - 1 ) // 2 + 1
        high_rnn_size = high_rnn_size * 128

        self.high_linear_in = nn.Linear(high_rnn_size, self.hidden_size)
        high_att_list = []
        for i in range(self.hb_att_layer):
            if self.hb_att_norm:
                high_att_list.append(nn.LayerNorm(self.hidden_size))
            high_att_list.append(
                GumbelMultiHeadedAttention(
                    1, self.hidden_size, self.d_feats // 2 + 1, self.K, self.hidden_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
        self.high_attention = nn.ModuleList(high_att_list)
        self.high_linear_out = nn.Linear(self.hidden_size, 960)

        low_rnn_size = self.d_feats // 2 + 1 + 4
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        # low_rnn_size = (low_rnn_size - (3 - 1) - 1 ) // 2 + 1
        low_rnn_size = low_rnn_size * 128
        
        self.low_linear_in = nn.Linear(low_rnn_size, self.hidden_size)
        low_att_list = []
        for i in range(self.hb_att_layer):
            if self.hb_att_norm:
                low_att_list.append(nn.LayerNorm(self.hidden_size))
            low_att_list.append(
                GumbelMultiHeadedAttention(
                    1, self.hidden_size, self.d_feats // 2 + 1, self.K, self.hidden_size, use_gumbel=self.use_gumbel, dropout_rate=self.dropout_rate
                ),
            )
        self.low_attention = nn.ModuleList(low_att_list)
        
        self.low_linear_out = nn.Linear(self.hidden_size, 960)

        # De-convolution 2d decoder
        self.decoder = torch.nn.ModuleDict({
            # 'deconv2d_1': GatedConvTranspose2dCell(in_channels=256, out_channels=128, kernel_size=(4, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_1': GatedConvTranspose2dCell(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_2': GatedConvTranspose2dCell(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_3': GatedConvTranspose2dCell(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate),
            'deconv2d_4': GatedConvTranspose2dCell(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=(1, 2), padding=0, dropout=self.dropout_rate, norm=False),
        })

        self.output_layer = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Linear(257, 257),
        )

    def forward(self, input, harmonic_bank):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        # input = torch.sqrt(input)
        input = torch.nn.functional.pad(input, (0, 0, 8, 0))
        # input = torch.log10(input + 1e-10)

        # input normalization
        input = torch.permute(input, (0, 2, 1))
        input = self.input_norm(input)
        input = torch.permute(input, (0, 2, 1))

        B, K, D = input.shape
        input = torch.unsqueeze(input, dim=1)
        hb_feats = torch.unsqueeze(harmonic_bank, dim=0).repeat(B, 1, 1)
        # input: (Batch, Channel=1, Timesteps, Frequency bin)
        # Chunk along feats
        input_low = input[:, :, :, :133]
        input_high = input[:, :, :, 124:]
        # input_low, input_high = torch.split(input, [129, 128], dim=-1)
        # input_high = torch.nn.functional.pad(input_high, (0, 1, 0, 0))

        # low encoder
        encoder_temp_low = [0]*len(self.low_encoder)
        i = 0 
        for k, conv in self.low_encoder.items():
            if i == 0:
                encoder_temp_low[i] = conv(input_low)
            else:
                encoder_temp_low[i] = conv(encoder_temp_low[i-1])
            i += 1
        encode_low_feats = encoder_temp_low[i-1]

        # Middle Layers
        # reshape: flatten along with channels
        B, C, K, D = encode_low_feats.shape
        encode_low_feats = torch.permute(encode_low_feats, (0, 2, 1, 3))
        encode_low_feats = torch.reshape(encode_low_feats, (B, K, C*D))
        
        encode_low_feats = self.low_linear_in(encode_low_feats)
        low_att_feats = encode_low_feats
        residual = encode_low_feats
        # for i in range(len(self.low_attention)):
            # if self.hb_att_norm:
            #     if isinstance(self.low_attention[i], MultiHeadedAttention):
            #         low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
            #     elif isinstance(self.low_attention[i], nn.LayerNorm): # layernorm
            #         low_att_feats = self.low_attention[i](low_att_feats + residual)
            #         residual = low_att_feats
            # else:
            #     low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
            #     low_att_feats = low_att_feats + residual
            #     residual = low_att_feats
        low_att_feats_list = []
        for i in range(len(self.low_attention)):
            if self.hb_att_norm:
                if isinstance(self.low_attention[i], MultiHeadedAttention):
                    low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
                    low_att_feats_list.append(low_att_feats)
                    if i != len(self.low_attention) - 1:
                        low_att_feats = low_att_feats + residual
                        residual = low_att_feats
                elif isinstance(self.low_attention[i], nn.LayerNorm): # layernorm
                    low_att_feats = self.low_attention[i](low_att_feats)
            else:
                low_att_feats = self.low_attention[i](low_att_feats, hb_feats, hb_feats, None)
                low_att_feats = low_att_feats + residual
                residual = low_att_feats
        # att_feats = self.attention[-1](att_feats, hb_feats, hb_feats, None)
        low_att_feats = torch.stack(low_att_feats_list, dim=0).sum(dim=0)
        low_att_feats = self.low_linear_out(low_att_feats)

        # High encoder
        encoder_temp_high = [0]*len(self.high_encoder)
        i = 0 
        for k, conv in self.high_encoder.items():
            if i == 0:
                encoder_temp_high[i] = conv(input_high)
            else:
                encoder_temp_high[i] = conv(encoder_temp_high[i-1])
            i += 1
        encode_high_feats = encoder_temp_high[i-1]

        B, C, K, D = encode_high_feats.shape
        encode_high_feats = torch.permute(encode_high_feats, (0, 2, 1, 3))
        encode_high_feats = torch.reshape(encode_high_feats, (B, K, C*D))

        encode_high_feats = self.high_linear_in(encode_high_feats)
        high_att_feats = encode_high_feats
        residual = encode_high_feats
        # for i in range(len(self.high_attention)):
        #     if self.hb_att_norm:
        #         if isinstance(self.high_attention[i], MultiHeadedAttention):
        #             high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
        #         elif isinstance(self.high_attention[i], nn.LayerNorm): # layernorm
        #             high_att_feats = self.high_attention[i](high_att_feats + residual)
        #             residual = high_att_feats
        #     else:
        #         high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
        #         high_att_feats = high_att_feats + residual
        #         residual = high_att_feats
        high_att_feats_list = []
        for i in range(len(self.high_attention)):
            if self.hb_att_norm:
                if isinstance(self.high_attention[i], MultiHeadedAttention):
                    high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
                    high_att_feats_list.append(high_att_feats)
                    if i != len(self.high_attention) - 1:
                        high_att_feats = high_att_feats + residual
                        residual = high_att_feats
                elif isinstance(self.high_attention[i], nn.LayerNorm): # layernorm
                    high_att_feats = self.high_attention[i](high_att_feats + residual)
            else:
                high_att_feats = self.high_attention[i](high_att_feats, hb_feats, hb_feats, None)
                high_att_feats = high_att_feats + residual
                residual = high_att_feats
        high_att_feats = torch.stack(high_att_feats_list, dim=0).sum(dim=0)
        high_att_feats = self.high_linear_out(high_att_feats)

        # # decoder_high_input = torch.reshape(rnn_high_feats, (B, K, C, D))
        decoder_input = torch.concat([low_att_feats, high_att_feats], dim=-1)
        # decoder_input = self.linear(att_feats)
        decoder_input = torch.reshape(decoder_input, (B, K, C, D*2+1))
        decoder_input = torch.permute(decoder_input, (0, 2, 1, 3))   
        
        i = len(self.decoder) - 1
        for k, deconv in self.decoder.items():
            if decoder_input.shape[-1] == 127:
                decoder_input = torch.nn.functional.pad(
                    decoder_input,
                    (0, 1, 0, 0)
                )
            # print(decoder_input.shape[-1], encoder_temp[i].shape[-1])
            # decoder_high_input = torch.concat((decoder_high_input, encoder_temp_high[i]), dim=1)
            decoder_input = deconv(decoder_input)
            i -= 1
        
        # enhanced_feats = torch.concat([decoder_low_input, decoder_high_input], dim=-1)
        # enhanced_feats = enhanced_feats[:, :, :, :-1]
        # enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = torch.squeeze(decoder_input, dim=1)
        enhanced_feats = self.output_layer(enhanced_feats)
        enhanced_feats = enhanced_feats[:, 8:, :]
        # enhanced_feats = 10 ** enhanced_feats
        # enhanced_feats = enhanced_feats ** 2 # magnitude to power

        return enhanced_feats
