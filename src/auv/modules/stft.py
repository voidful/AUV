import torch
from torch import nn


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center=True,
    ):
        super().__init__()
        self.center = center
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T * hop_length)

        if not self.center:
            pad = self.win_length - self.hop_length
            x = nn.functional.pad(x, (pad // 2, pad // 2), mode="reflect")

        stft_spec = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            onesided=True,
            return_complex=True,
        )
        stft_spec = torch.view_as_real(stft_spec)  # (B, n_fft // 2 + 1, T, 2)

        rea = stft_spec[:, :, :, 0]  # (B, n_fft // 2 + 1, T, 2)
        imag = stft_spec[:, :, :, 1]  # (B, n_fft // 2 + 1, T, 2)

        log_mag = torch.log(
            torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2))) + 1e-5
        )  # (B, n_fft // 2 + 1, T)
        phase = torch.atan2(imag, rea)  # (B, n_fft // 2 + 1, T)

        return log_mag, phase


class STFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, **kwargs):
        super().__init__()
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=False)
        inp_dim = n_fft + 2
        self.inp = nn.Linear(inp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_mag, phase = self.stft(x)
        x = torch.cat((log_mag, phase), dim=1).transpose(1, 2)
        x = self.inp(x)  # (B, T, C)
        return x


class ISTFT(nn.Module):
    # https://github.com/pytorch/pytorch/issues/62323
    # fix padding type to "same" here

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        pad = (self.win_length - self.hop_length) // 2

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, **kwargs):
        super().__init__()
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio
