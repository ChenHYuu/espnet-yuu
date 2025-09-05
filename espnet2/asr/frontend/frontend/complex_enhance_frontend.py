import copy
import math
from typing import Optional, OrderedDict, Tuple, Union
import librosa
import gc

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

class FDCUIrmFrontend(AbsFrontend):
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
        type = 'fdcu_v1',
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

        if type == 'fdcu_v1':
            self.build_model_v1()
            self.enhance = self.enhance_v1
        elif type == 'fdcu_v2':
            self.build_model_v2()
            self.enhance = self.enhance_v2

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

    def build_model_v1(self):
        self.d_feats = self.n_fft // 2 + 1
        self.model = FDCUnet(self.d_feats, self.dropout_rate)
        self.irm_sigmoid = nn.Sigmoid() 
        # self.latency = self.model.output_latency()
        self.latency = 0

    def enhance_v1(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        # input = torch.sqrt(input)
        # print(f"input size: {input.shape}")
        input = torch.nn.functional.pad(input, (0, 0, 0, 0, self.latency, 0))
        # print(f"input after padding: {input.shape}")
        input_feats = torch.unsqueeze(input, dim=1)
        # gc.collect()
        # print(f"before model:{torch.cuda.list_gpu_processes()}")
        magn_out, phase_out = self.model(input_feats)
        # gc.collect()
        # print(f"after model:{torch.cuda.list_gpu_processes()}")

        magn_out = torch.squeeze(magn_out, dim=1)
        phase_out = torch.squeeze(phase_out, dim=1)

        magn_out_real = magn_out[..., 0] 
        magn_out_imag = magn_out[..., 1]
        # Small epsilon to avoid negative values in sqrt due to precision
        irm_out = torch.sqrt(magn_out_real ** 2 + magn_out_imag ** 2 + 1e-8)
        irm_out = self.irm_sigmoid(irm_out)

        input_real = input[..., 0]
        input_imag = input[..., 1]
        magn_src = torch.sqrt(input_real ** 2 + input_imag ** 2)

        magn_spectrogram = irm_out * magn_src

        phase_out_real = phase_out[..., 0]
        phase_out_imag = phase_out[..., 1]
        phase_complex = torch.complex(phase_out_real, phase_out_imag)
        phase_spectrogram = torch.angle(phase_complex)

        # print(f"output size: {magn_spectrogram.shape}, {phase_spectrogram.shape}")
        magn_spec = magn_spectrogram[:, self.latency:, :]
        phase_spec = phase_spectrogram[:, self.latency:, :]
        # print(f"output size after prume: {magn_spec.shape}, {phase_spec.shape}")
        # gc.collect()
        # print(f"after enhance:{torch.cuda.list_gpu_processes()}")
        return magn_spec, phase_spec
    
    def build_model_v2(self):
        self.d_feats = self.n_fft // 2 + 1
        self.channels = [16, 16, 32, 32, 32]
        self.kernels  = [(7,1), (7,1), (7,5), (7,5), (7,5)]
        self.strides  = [(1,1), (1,1), (1,2), (1,2), (1,2)]
        skip_factor = 0
        self.model = FDCUBlock(input_size=self.d_feats, 
                               channels=self.channels, 
                               kernels=self.kernels, 
                               strides=self.strides, 
                               rnn_layers=1, 
                               rnn_hidden_size=128, 
                               skip_factor=skip_factor,
                               dropout_rate=self.dropout_rate, 
                               stage=2,
                              )
        # self.latency = self.model.output_latency()
        self.latency = 0

    def enhance_v2(self, input):
        # 6. Power spectrum -> ehancement network -> enhanced power spectrum
        # encode -> reshape -> lstm -> reshape -> decode
        # input: (Batch, Timesteps, frequency bin)
        input = torch.nn.functional.pad(input, (0, 0, 0, 0, self.latency, 0))
        input_feats = torch.unsqueeze(input, dim=1)
        stft_out = self.model(input_feats)
        stft_out = torch.squeeze(stft_out, dim=1)

        stft_out_real = stft_out[..., 0] 
        stft_out_imag = stft_out[..., 1]

        # Small epsilon to avoid negative values in sqrt due to precision
        magn_spectrogram = torch.sqrt(stft_out_real ** 2 + stft_out_imag ** 2 + 1e-8)

        phase_complex = torch.complex(stft_out_real, stft_out_imag)
        phase_spectrogram = torch.angle(phase_complex)

        magn_spec = magn_spectrogram[:, self.latency:, :]
        phase_spec = phase_spectrogram[:, self.latency:, :]
        return magn_spec, phase_spec

    def build_model_v3(self):
        self.d_feats = self.n_fft // 2 + 1
        self.channels = [16, 16, 32, 32, 32]
        self.kernels  = [(7,1), (7,1), (7,5), (7,5), (7,5)]
        self.strides  = [(1,1), (1,1), (1,2), (1,2), (1,2)]
        skip_factor = 0
        self.model = FDCUBlock(input_size=self.d_feats, 
                               channels=self.channels, 
                               kernels=self.kernels, 
                               strides=self.strides, 
                               rnn_layers=1, 
                               rnn_hidden_size=128, 
                               skip_factor=skip_factor,
                               dropout_rate=self.dropout_rate, 
                               stage=2,
                              )
        # self.latency = self.model.output_latency()
        self.latency = 0

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # torch.autograd.set_detect_anomaly(True)
        # input_power_clean = None
        # print(f"wav length:{input_lengths}")
        # input_wav_clean = None
        if self.return_spec:
            input_wav_clean = input

        if self.add_awgn and self.training:
            input = add_awgn_batch(input, self.snr)
            
        input_stft, feats_lens = self._compute_stft(input, input_lengths, ret_list=True)
        enhacne_magn, enhance_phase = self.enhance(input_stft)
        enhanced_wav, enhanced_wav_lens = self._to_wavform(enhacne_magn, enhance_phase, input_lengths)

        if self.return_spec:
            feats_intermediate = (input_wav_clean, enhanced_wav)

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

        # torch.cuda.empty_cache()

        if self.return_spec:
            return input_feats, feats_lens, feats_intermediate
        else:
            return input_feats, feats_lens
        
    def _to_wavform(self, input_magn, input_phase, input_lens):
        cos = torch.cos(input_phase)
        sin = torch.sin(input_phase)
        norm = torch.complex(cos, sin)
        
        spec_with_phase = input_magn * norm
        # print(f"spectrogram size: {spec_with_phase.shape} {input_lens}")
        wav, wav_lens = self.stft.inverse(spec_with_phase, input_lens)
        # print(f"wav size: {wav.shape}, {wav_lens.shape}")
        # wav = self._unitize(wav)
        wav = nn.functional.tanh(wav)
        # if not self.training:
        #     wav = nn.functional.softshrink(wav, lambd=0.2)
        del spec_with_phase, cos, sin, norm
        return wav, wav_lens

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor, ret_list=False,
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        if ret_list:
            return input_stft, feats_lens 
        else:
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

class FDCUnet(nn.Module):
    def __init__(self, input_size, skip_factor=0, dropout_rate=0.0):
        super(FDCUnet, self).__init__()
        self.input_size = input_size
        self.channels = [16, 16, 32, 32, 32]
        self.kernels  = [(7,1), (7,1), (7,5), (7,5), (7,5)]
        self.strides  = [(1,1), (1,1), (1,2), (1,2), (1,2)]
        skip_factor = 1 
        self.dropout_rate = dropout_rate

        self.extract_block = FDCUBlock(input_size=self.input_size, 
                                       channels=self.channels, 
                                       kernels=self.kernels, 
                                       strides=self.strides, 
                                       rnn_layers=1, 
                                       rnn_hidden_size=128, 
                                       skip_factor=skip_factor,
                                       dropout_rate=self.dropout_rate, 
                                       stage=1
                                      )
        
        stage2_input_size = self.extract_block.output_wdim()

        self.magnitude_block = FDCUBlock(input_size=stage2_input_size, 
                                         channels=self.channels, 
                                         kernels=self.kernels, 
                                         strides=self.strides, 
                                         rnn_layers=1, 
                                         rnn_hidden_size=128, 
                                         skip_factor=skip_factor,
                                         dropout_rate=self.dropout_rate, 
                                         stage=2,
                                        )
        self.phase_block     = FDCUBlock(input_size=stage2_input_size, 
                                         channels=self.channels, 
                                         kernels=self.kernels, 
                                         strides=self.strides, 
                                         rnn_layers=1, 
                                         rnn_hidden_size=128, 
                                         skip_factor=skip_factor,
                                         dropout_rate=self.dropout_rate, 
                                         stage=2,
                                        )   
    
    def forward(self, x):
        # gc.collect()
        # print(f"before extract:{torch.cuda.list_gpu_processes()}")
        magn_in, phase_in = self.extract_block(x)
        # gc.collect()
        # print(f"after extract:{torch.cuda.list_gpu_processes()}")
        magn_out = self.magnitude_block(magn_in)
        # gc.collect()
        # print(f"after magn:{torch.cuda.list_gpu_processes()}")
        # torch.cuda.empty_cache()

        phase_out = self.phase_block(phase_in)
        # gc.collect()
        # print(f"after phase:{torch.cuda.list_gpu_processes()}")
        torch.cuda.empty_cache()


        return magn_out, phase_out

    def output_latency(self):
        start = 100
        dims = self.extract_block.output_hdim(start)
        dims = self.magnitude_block.output_hdim(dims)
        return start - dims

class minFDCUnet(nn.Module):
    def __init__(self, input_size, skip_factor=0, dropout_rate=0.0):
        super(minFDCUnet, self).__init__()
        self.input_size = input_size
        self.channels = [16, 16, 32, 32, 32]
        self.kernels  = [(7,1), (7,1), (7,5), (7,5), (7,5)]
        self.strides  = [(1,1), (1,1), (1,2), (1,2), (1,2)]
        skip_factor = 1 
        self.dropout_rate = dropout_rate
        self.skip_factor = skip_factor
        rnn_hidden_size=128
        rnn_layers=1

        # Prepare encoder output channels, kernels, strides
        encoder_out_channels = self.channels.copy()
        encoder_kernels = self.kernels.copy()
        encoder_strides = self.strides.copy()

        # Calculate decoder input channels, kernels, strides
        decoder_in_channels = torch.tensor(self.channels)
        decoder_in_channels[self.skip_factor::self.skip_factor+1] *= 2
        decoder_in_channels = decoder_in_channels.tolist()
        decoder_in_channels.reverse()
        decoder_kernels = self.kernels.copy()
        decoder_kernels.reverse()
        decoder_strides = self.strides.copy()
        decoder_strides.reverse()

        self.encoder = cEncoderBlock(out_channels=encoder_out_channels, 
                                     kernels=encoder_kernels, 
                                     strides=encoder_strides, 
                                     skip_factor=self.skip_factor, 
                                     dropout_rate=self.dropout_rate)

        # RNN
        final_out_channel = self.channels[-1]
        rnn_input_size = self.encoder.output_wdim(input_size) * final_out_channel # Get feature dims of encoder output
        # self.rnn     = cLSTMBlock(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, projection_dim=rnn_input_size)
        self.extract_rnn     = cMinGruBlock(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, projection_dim=rnn_input_size)
        self.mag_rnn     = cMinGruBlock(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, projection_dim=rnn_input_size)
        self.phase_rnn     = cMinGruBlock(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, projection_dim=rnn_input_size)

        
        stage2_input_size = self.extract_block.output_wdim()

        self.magnitude_block = FDCUBlock(input_size=stage2_input_size, 
                                         channels=self.channels, 
                                         kernels=self.kernels, 
                                         strides=self.strides, 
                                         rnn_layers=1, 
                                         rnn_hidden_size=128, 
                                         skip_factor=skip_factor,
                                         dropout_rate=self.dropout_rate, 
                                         stage=2,
                                        )
        self.phase_block     = FDCUBlock(input_size=stage2_input_size, 
                                         channels=self.channels, 
                                         kernels=self.kernels, 
                                         strides=self.strides, 
                                         rnn_layers=1, 
                                         rnn_hidden_size=128, 
                                         skip_factor=skip_factor,
                                         dropout_rate=self.dropout_rate, 
                                         stage=2,
                                        )   

class FDCUBlock(nn.Module):
    def __init__(
        self, 
        input_size, 
        channels: list, 
        kernels: list, 
        strides: list, 
        rnn_layers=1, 
        rnn_hidden_size=128, 
        skip_factor:int=1,
        dropout_rate=0,
        stage=1,
    ):   
        super(FDCUBlock, self).__init__()
        self.channels     = channels
        self.kernels      = kernels
        self.strides      = strides
        self.stage        = stage
        self.skip_factor  = skip_factor

        self.input_size = input_size

        # Prepare encoder output channels, kernels, strides
        encoder_out_channels = channels.copy()
        encoder_kernels = kernels.copy()
        encoder_strides = strides.copy()

        # Calculate decoder input channels, kernels, strides
        decoder_in_channels = torch.tensor(channels)
        decoder_in_channels[self.skip_factor::self.skip_factor+1] *= 2
        decoder_in_channels = decoder_in_channels.tolist()
        decoder_in_channels.reverse()
        decoder_kernels = kernels.copy()
        decoder_kernels.reverse()
        decoder_strides = strides.copy()
        decoder_strides.reverse()

        # Encoder
        self.encoder = cEncoderBlock(out_channels=encoder_out_channels, kernels=encoder_kernels, strides=encoder_strides, skip_factor=self.skip_factor, dropout_rate=dropout_rate)
        
        # RNN
        final_out_channel = self.channels[-1]
        rnn_input_size = self.encoder.output_wdim(input_size) * final_out_channel # Get feature dims of encoder output
        # self.rnn     = cLSTMBlock(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, projection_dim=rnn_input_size)
        self.rnn     = cMinGruBlock(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_layers, projection_dim=rnn_input_size)

        # Decoders
        # Reference: https://www.isca-archive.org/interspeech_2021/sun21_interspeech.pdf
        if self.stage == 1:
            self.decoder1 = cDecoderBlock(in_channels=decoder_in_channels, kernels=decoder_kernels, strides=decoder_strides, skip_factor=self.skip_factor, dropout_rate=dropout_rate)
            self.decoder2 = cDecoderBlock(in_channels=decoder_in_channels, kernels=decoder_kernels, strides=decoder_strides, skip_factor=self.skip_factor, dropout_rate=dropout_rate)
        elif self.stage == 2:
            self.decoder1 = cDecoderBlock(in_channels=decoder_in_channels, kernels=decoder_kernels, strides=decoder_strides, skip_factor=self.skip_factor, dropout_rate=dropout_rate)    

    def forward(self, x):
        # 1. Encode
        encoder_out, encoder_outs, encoder_outs_dim = self.encoder(x)
        
        # 2. RNN
        # Reshape to fit rnn input
        B, C, K, D, F = encoder_out.shape
        # Maybe problem here
        encoder_out = torch.permute(encoder_out, (0, 2 ,1, 3, 4))
        encoder_out = torch.reshape(encoder_out, (B, K, C*D, F))
        # print(encoder_out.shape)
        rnn_out = self.rnn(encoder_out)
        # Reshape to fit decoder input        
        rnn_out = torch.reshape(rnn_out, (B, K, C, D, F))
        rnn_out = torch.permute(rnn_out, (0, 2, 1, 3, 4))

        if self.stage == 1:
            decoder1_out = self.decoder1(rnn_out, encoder_outs, encoder_outs_dim)
            decoder2_out = self.decoder2(rnn_out, encoder_outs, encoder_outs_dim)
            del encoder_outs, rnn_out, encoder_out
            return decoder1_out, decoder2_out
        elif self.stage == 2:
            decoder1_out = self.decoder1(rnn_out, encoder_outs, encoder_outs_dim)
            del encoder_outs, rnn_out, encoder_out
            return decoder1_out
            
        return None
    
    def output_wdim(self):
        dims = self.encoder.output_wdim(self.input_size)
        dims = self.decoder1.output_wdim(dims)
        return dims
    
    def output_hdim(self, h):
        dims = h
        dims = self.encoder.output_hdim(dims)
        dims = self.decoder1.output_hdim(dims)
        return dims

class cEncoderBlock(nn.Module):
    def __init__(self, out_channels, kernels, strides, skip_factor, dropout_rate=0):
        super(cEncoderBlock, self).__init__()
       
        self.out_channels = out_channels
        self.kernels      = kernels
        self.strides      = strides
        self.skip_factor  = skip_factor
        
        self.layers = len(self.out_channels)

        cnns = []
        in_channel = 1
        for n in range(self.layers):
            out_channel = self.out_channels[n]
            kernel = self.kernels[n]
            stride = self.strides[n]
            cnns.append(cEncoderCell(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=stride, dropout_rate=dropout_rate))
            in_channel = out_channel
        
        self.cells = nn.ModuleList(cnns)
        del cnns
    
    def forward(self, x):
        encoder_outs = []
        encoder_outs_dim = []
        encoder_in = x
        for n, cell in enumerate(self.cells):
            encoder_out = cell(encoder_in)
            if (n+1) % (self.skip_factor+1) == 0:
                # Not skip layer
                encoder_outs.append(encoder_out)
            else:
                # Is skip layer
                encoder_outs.append(None)
            encoder_outs_dim.append(encoder_out.shape[-2])
            encoder_in = encoder_out
        
        del encoder_in
        # Reverse encoder output before input to decoder
        encoder_outs.reverse()
        encoder_outs_dim.reverse()
        return encoder_out, encoder_outs, encoder_outs_dim

    def output_wdim(self, d_feats):
        dims = d_feats
        for n in range(self.layers):
            dims = math.floor((dims - (self.kernels[n][1] -1) - 1) / self.strides[n][1] + 1)
        return dims
    
    def output_hdim(self, d_feats):
        dims = d_feats
        for n in range(self.layers):
            dims = math.floor((dims - (self.kernels[n][0] -1) - 1) / self.strides[n][0] + 1)
        return dims

class cDecoderBlock(nn.Module):
    def __init__(self, in_channels, kernels, strides, skip_factor, dropout_rate=0):
        super(cDecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.kernels     = kernels
        self.strides     = strides
        self.skip_factor = skip_factor

        self.layers = len(self.kernels)

        cnns = []
        out_channel = 1
        self.in_channels.append(int(1))
        for n in range(0, self.layers):
            in_channel = self.in_channels[n]
            out_channel = self.in_channels[n+1]
            kernel = self.kernels[n]
            stride = self.strides[n]
            if (self.layers-n+1) % (self.skip_factor+1) == 0:
                # Is skip layer
                cnns.append(cDecoderCell(in_channel, max(out_channel//2, 1), kernel, stride, dropout_rate=dropout_rate))
            else:
                # Not skip layer
                cnns.append(cDecoderCell(in_channel, out_channel, kernel, stride, dropout_rate=dropout_rate))
        # for n in range(self.layers, 0, -1):
        #     in_channel = self.in_channels[n-1]
        #     kernel = self.kernels[n-1]
        #     stride = self.strides[n-1]
        #     print(n, in_channel, out_channel)
        #     cnns.append(cDecoderCell(in_channel, out_channel, kernel, stride, dropout_rate=dropout_rate))
        #     if (self.layers-n) % (self.skip_factor+1) == 0:
        #         # Not skip layer
        #         out_channel = in_channel // 2
        #     else:
        #         # Is skip layer
        #         out_channel = in_channel
        
        # cnns.reverse()
        self.cells = nn.ModuleList(cnns)
        del cnns
    
    def forward(self, x: torch.Tensor, encoder_outs, encoder_outs_dim):
        decoder_in = x
        for n, cell in enumerate(self.cells):
            # Pad decoder input to be same size of encoder output
            if decoder_in.shape[-2] != encoder_outs_dim[n]:
                # print(f"decoder in before: {decoder_in.shape}")
                decoder_in = torch.nn.functional.pad(
                    decoder_in,
                    (0, 0, 0, 1, 0, 0)
                )
                # print(f"decoder in after: {decoder_in.shape}")
            if encoder_outs[n] != None:
                decoder_in = torch.concat([decoder_in, encoder_outs[n]], dim=1)
            decoder_out = cell(decoder_in)
            decoder_in = decoder_out
        del decoder_in
        return decoder_out
    
    def output_wdim(self, d_feats):
        dims = d_feats
        for n in range(self.layers):
            dims = (dims - 1) * self.strides[n][1] + (self.kernels[n][1] - 1) + 1
        return dims
    
    def output_hdim(self, d_feats):
        dims = d_feats
        for n in range(self.layers):
            dims = (dims - 1) * self.strides[n][0] + (self.kernels[n][0] - 1) + 1
        return dims

class cEncoderCell(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dropout_rate=0):
            super(cEncoderCell, self).__init__()
            self.cnn = cConv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.norm = CBatchNorm2d(out_channels)
            self.activation = cPReLU()
            self.dropout = nn.Dropout(dropout_rate)
        
        def forward(self, x):
            # conv = self.cnn(x)
            # norm = self.norm(conv)
            # actv = self.activation(norm)
            # output = self.dropout(actv)
            return self.dropout(self.activation(self.norm(self.cnn(x))))
        
class cDecoderCell(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0, dropout_rate=0):
            super(cDecoderCell, self).__init__()
            self.cnn = cConvTranspose2d(in_channels, out_channels, kernel_size, stride, output_padding, padding)
            self.norm = CBatchNorm2d(out_channels)
            self.activation = cPReLU()
            self.dropout = nn.Dropout(dropout_rate)
        
        def forward(self, x):
            # conv = self.cnn(x)
            # norm = self.norm(conv)
            # actv = self.activation(norm)
            # output = self.dropout(actv)
            return self.dropout(self.activation(self.norm(self.cnn(x))))
        
class cMinGruBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, projection_dim=None, bidirectional=False, batch_first=True, dropout_rate=0):
        super(cMinGruBlock, self).__init__()
        self.rnn = cMinGru(input_size, hidden_size, num_layers=num_layers, projection_dim=projection_dim, bidirectional=bidirectional, batch_first=batch_first)
        self.norm = cLayerNorm2d(projection_dim)
        self.activation = cPReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # # print(f"lstm in: {x.shape}")
        # rnn = self.rnn(x)
        # # print(f"rnn: {rnn.shape}")
        # norm = self.norm(rnn)
        # actv = self.activation(norm)
        # output = self.dropout(actv)
        return self.dropout(self.activation(self.norm(self.rnn(x))))


class cLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, projection_dim=None, bidirectional=False, batch_first=True, dropout_rate=0):
        super(cLSTMBlock, self).__init__()
        self.rnn = cLSTM(input_size, hidden_size, num_layers=num_layers, projection_dim=projection_dim, bidirectional=bidirectional, batch_first=batch_first)
        self.norm = cLayerNorm2d(input_size)
        self.activation = cPReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # # print(f"lstm in: {x.shape}")
        # y = self.rnn(x)
        # # print(f"rnn: {rnn.shape}")
        # y = self.norm(y)
        # y = self.activation(y)
        # y = self.dropout(y)
        return self.dropout(self.activation(self.norm(self.rnn(x))))

class cConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        return torch.stack([c_real, c_im], dim=-1)

class cConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
        
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        # print(x.shape)
        x_real = x[..., 0]
        x_im = x[..., 1]
        # print(x_real.shape)
        
        c_real = self.real_convt(x_real) - self.im_convt(x_im)
        c_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        return torch.stack([c_real, c_im], dim=-1)
    
class cLayerNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True):
        super().__init__()
        
        self.real_b = nn.LayerNorm(normalized_shape, eps, elementwise_affine, bias)
        self.im_b = nn.LayerNorm(normalized_shape, eps, elementwise_affine, bias) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        x_real = self.real_b(x_real)
        x_im = self.im_b(x_im)  
        
        return torch.stack([x_real, x_im], dim=-1)

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        x_real = self.real_b(x_real)
        x_im = self.im_b(x_im)  
        
        return torch.stack([x_real, x_im], dim=-1)

# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch 
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55
# class cBatchNorm(torch.nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
#             track_running_stats=True, complex_axis=1):
#         super(cBatchNorm, self).__init__()
#         self.num_features        = num_features
#         self.eps                 = eps
#         self.momentum            = momentum
#         self.affine              = affine
#         self.track_running_stats = track_running_stats 
        
#         self.complex_axis = complex_axis

#         if self.affine:
#             self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
#             self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
#             self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
#             self.Br  = torch.nn.Parameter(torch.Tensor(self.num_features))
#             self.Bi  = torch.nn.Parameter(torch.Tensor(self.num_features))
#         else:
#             self.register_parameter('Wrr', None)
#             self.register_parameter('Wri', None)
#             self.register_parameter('Wii', None)
#             self.register_parameter('Br',  None)
#             self.register_parameter('Bi',  None)
        
#         if self.track_running_stats:
#             self.register_buffer('RMr',  torch.zeros(self.num_features))
#             self.register_buffer('RMi',  torch.zeros(self.num_features))
#             self.register_buffer('RVrr', torch.ones (self.num_features))
#             self.register_buffer('RVri', torch.zeros(self.num_features))
#             self.register_buffer('RVii', torch.ones (self.num_features))
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('RMr',                 None)
#             self.register_parameter('RMi',                 None)
#             self.register_parameter('RVrr',                None)
#             self.register_parameter('RVri',                None)
#             self.register_parameter('RVii',                None)
#             self.register_parameter('num_batches_tracked', None)
#         self.reset_parameters()

#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.RMr.zero_()
#             self.RMi.zero_()
#             self.RVrr.fill_(1)
#             self.RVri.zero_()
#             self.RVii.fill_(1)
#             self.num_batches_tracked.zero_()

#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             self.Br.data.zero_()
#             self.Bi.data.zero_()
#             self.Wrr.data.fill_(1)
#             self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
#             self.Wii.data.fill_(1)

#     def _check_input_dim(self, xr, xi):
#         assert(xr.shape == xi.shape)
#         assert(xr.size(1) == self.num_features)

#     def forward(self, x):
#         #self._check_input_dim(xr, xi)
        
#         xr = x[..., 0]
#         xi = x[..., 1]
#         exponential_average_factor = 0.0

#         if self.training and self.track_running_stats:
#             self.num_batches_tracked += 1
#             if self.momentum is None:  # use cumulative moving average
#                 exponential_average_factor = 1.0 / self.num_batches_tracked.item()
#             else:  # use exponential moving average
#                 exponential_average_factor = self.momentum

#         #
#         # NOTE: The precise meaning of the "training flag" is:
#         #       True:  Normalize using batch   statistics, update running statistics
#         #              if they are being collected.
#         #       False: Normalize using running statistics, ignore batch   statistics.
#         #
#         training = self.training or not self.track_running_stats
#         redux = [i for i in reversed(range(xr.dim())) if i!=1]
#         vdim  = [1] * xr.dim()
#         vdim[1] = xr.size(1)

#         #
#         # Mean M Computation and Centering
#         #
#         # Includes running mean update if training and running.
#         #
#         if training:
#             Mr, Mi = xr, xi
#             for d in redux:
#                 Mr = Mr.mean(d, keepdim=True)
#                 Mi = Mi.mean(d, keepdim=True)
#             if self.track_running_stats:
#                 self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
#                 self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
#         else:
#             Mr = self.RMr.view(vdim)
#             Mi = self.RMi.view(vdim)
#         xr, xi = xr-Mr, xi-Mi

#         #
#         # Variance Matrix V Computation
#         #
#         # Includes epsilon numerical stabilizer/Tikhonov regularizer.
#         # Includes running variance update if training and running.
#         #
#         if training:
#             Vrr = xr * xr
#             Vri = xr * xi
#             Vii = xi * xi
#             for d in redux:
#                 Vrr = Vrr.mean(d, keepdim=True)
#                 Vri = Vri.mean(d, keepdim=True)
#                 Vii = Vii.mean(d, keepdim=True)
#             if self.track_running_stats:
#                 self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
#                 self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
#                 self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
#         else:
#             Vrr = self.RVrr.view(vdim)
#             Vri = self.RVri.view(vdim)
#             Vii = self.RVii.view(vdim)
#         Vrr   = Vrr + self.eps
#         Vri   = Vri
#         Vii   = Vii + self.eps

#         #
#         # Matrix Inverse Square Root U = V^-0.5
#         #
#         # sqrt of a 2x2 matrix,
#         # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
#         tau   = Vrr + Vii
#         delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
#         s     = delta.sqrt()
#         t     = (tau + 2*s).sqrt()

#         # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
#         rst   = (s * t).reciprocal()
#         Urr   = (s + Vii) * rst
#         Uii   = (s + Vrr) * rst
#         Uri   = (  - Vri) * rst

#         #
#         # Optionally left-multiply U by affine weights W to produce combined
#         # weights Z, left-multiply the inputs by Z, then optionally bias them.
#         #
#         # y = Zx + B
#         # y = WUx + B
#         # y = [Wrr Wri][Urr Uri] [xr] + [Br]
#         #     [Wir Wii][Uir Uii] [xi]   [Bi]
#         #
#         if self.affine:
#             Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
#             Zrr = (Wrr * Urr) + (Wri * Uri)
#             Zri = (Wrr * Uri) + (Wri * Uii)
#             Zir = (Wri * Urr) + (Wii * Uri)
#             Zii = (Wri * Uri) + (Wii * Uii)
#         else:
#             Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

#         yr = (Zrr * xr) + (Zri * xi)
#         yi = (Zir * xr) + (Zii * xi)

#         if self.affine:
#             yr = yr + self.Br.view(vdim)
#             yi = yi + self.Bi.view(vdim)

#         output = torch.stack([yr, yi], dim=-1)
#         return output

#     def extra_repr(self):
#         return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
#                 'track_running_stats={track_running_stats}'.format(**self.__dict__) 

class cPReLU(nn.Module):

    def __init__(self):
        super(cPReLU, self).__init__()
        self.r_prelu = nn.PReLU()        
        self.i_prelu = nn.PReLU()

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        x_real = self.r_prelu(x_real)
        x_im = self.i_prelu(x_im)
        
        return torch.stack([x_real, x_im], dim=-1)

class cLSTM(nn.Module):
    """
    Class of complex valued LSTM Layer
    """
    def __init__(self, input_size, hidden_size, num_layers=1, projection_dim=None, bidirectional=False, batch_first=True):
        super(cLSTM, self).__init__()

        self.input_dim = input_size
        self.rnn_units = hidden_size
        # print(input, hidden_size)
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
        if bidirectional:
            bidirectional=2
        else:
            bidirectional=1
        if projection_dim is not None:
            self.projection_dim = projection_dim
            self.r_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, x):
        # print(f"x: {x.shape}")
        x_real = x[..., 0]
        x_im = x[..., 1]
        # print(f"x_r: {x_real.shape}")

        r2r_out = self.real_lstm(x_real)[0]
        r2i_out = self.imag_lstm(x_real)[0]
        i2r_out = self.real_lstm(x_im)[0]
        i2i_out = self.imag_lstm(x_im)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        #print(real_out.shape,imag_out.shape)
        # output = torch.stack([real_out, imag_out], dim=-1)
        return torch.stack([real_out, imag_out], dim=-1)
    
    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()

class cMinGru(nn.Module):
    """
    Class of complex valued LSTM Layer
    """
    def __init__(self, input_size, hidden_size=128, num_layers=1, projection_dim=None, bidirectional=False, batch_first=True):
        super(cMinGru, self).__init__()

        self.input_dim = input_size
        self.rnn_units = hidden_size
        self.r_proj = nn.Linear(self.input_dim, self.rnn_units)
        self.i_proj = nn.Linear(self.input_dim, self.rnn_units)
        self.real_gru = minGRU(self.rnn_units)
        self.imag_gru = minGRU(self.rnn_units)

        if projection_dim is not None:
            self.projection_dim = projection_dim
            self.r_trans = nn.Linear(self.rnn_units, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, x):
        # print(f"x: {x.shape}")
        x_real = x[..., 0]
        x_im = x[..., 1]
        # print(f"x_r: {x_real.shape}")

        x_real = self.r_proj(x_real)
        x_im = self.i_proj(x_im)

        r2r_out = self.real_gru(x_real)
        r2i_out = self.imag_gru(x_real)
        i2r_out = self.real_gru(x_im)
        i2i_out = self.imag_gru(x_im)
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        #print(real_out.shape,imag_out.shape)
        output = torch.stack([real_out, imag_out], dim=-1)
        return output

# class cDropout(nn.Module):
#     def __init__(self, dropout_rate):
#         self.real_dropout = nn.Dropout(dropout_rate)
#         self.imag_dropout = nn.Dropout(dropout_rate)
    
#     def forward(self, x):
#         x_real = x[..., 0]
#         x_im = x[..., 1]

#         n_real = self.real_dropout(x_real)
#         n_im = self.imag_dropout(x_im)
#         output = torch.stack([n_real, n_im], dim=-1)
#         return output

def S_SISNRLoss(x:torch.Tensor, s:torch.Tensor, eps=1e-8):
    """
    calculate training loss
    input:
        x: separated signal, (B, T) tensor
        s: reference signal, (B, T) tensor
    return:
        sisnr: (B, ) tensor
    """
    
    if x.shape != s.shape:
        raise RuntimeError(
            f"Dimension mismatch when calculate si-snr, {x.shape} vs {s.shape}"
        )

    x_zm = (x - torch.mean(x, dim=-1, keepdim=True)).flatten()
    s_zm = (s - torch.mean(s, dim=-1, keepdim=True)).flatten()

    inner_prod = torch.sum(x_zm * s_zm, dim=-1)
    x_norm = torch.sqrt(torch.sum(x_zm ** 2, dim=-1))
    s_norm = torch.sqrt(torch.sum(s_zm ** 2, dim=-1))
    cos = inner_prod / (x_norm * s_norm)
    
    # 10 * torch.log10((1 + cos)/(1 - cos)) take nagetvie loss
    loss = 10 * torch.log10((1 - cos)/(1 + cos)) + 65 # add 65 to avoid to negative loss
    return loss


