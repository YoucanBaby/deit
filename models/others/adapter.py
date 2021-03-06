import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat


class InputAdapter(nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels


class FourierInputAdapter(InputAdapter):
    def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands = 64):
        *self.spatial_shape, num_image_channels = image_shape
        self.image_shape = image_shape
        self.num_frequency_bands = num_frequency_bands

        super().__init__(num_input_channels=num_image_channels + self._num_position_encoding_channels())

        # create encodings for single example
        pos = self._positions()
        enc = self._position_encodings(pos)

        # flatten encodings along spatial dimensions
        enc = rearrange(enc, "... c -> (...) c")

        # position encoding prototype
        self.register_buffer("position_encoding", enc)

    def _positions(self, v_min=-1.0, v_max=1.0):
        """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].

        :param v_min: minimum coordinate value per dimension.
        :param v_max: maximum coordinate value per dimension.
        :return: position coordinates tensor of shape (*shape, len(shape)).
        """
        coords = [torch.linspace(v_min, v_max, steps=s) for s in self.spatial_shape]
        return torch.stack(torch.meshgrid(*coords), dim=len(self.spatial_shape))

    def _position_encodings(
        self, p: torch.Tensor, max_frequencies: Optional[Tuple[int, ...]] = None, include_positions: bool = True
    ) -> torch.Tensor:
        """Fourier-encode positions p using self.num_bands frequency bands.

        :param p: positions of shape (*d, c) where c = len(d).
        :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
               2-tuple for images, ...). If `None` values are derived from shape of p.
        :param include_positions: whether to include input positions p in returned encodings tensor.
        :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
        """
        encodings = []

        if max_frequencies is None:
            max_frequencies = p.shape[:-1]

        frequencies = [
            torch.linspace(1.0, max_freq / 2.0, self.num_frequency_bands, device=p.device)
            for max_freq in max_frequencies
        ]
        frequency_grids = []

        for i, frequencies_i in enumerate(frequencies):
            frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

        if include_positions:
            encodings.append(p)

        encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
        encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

        return torch.cat(encodings, dim=-1)

    def _num_position_encoding_channels(self, include_positions: bool = True) -> int:
        # 2 * (2 * 64 + 1)
        return len(self.spatial_shape) * (2 * self.num_frequency_bands + include_positions)

    def forward(self, x):
        b, *d = x.shape

        # (h, w, c)
        if tuple(d) != self.image_shape:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

        # repeat position encoding along batch dimension
        x_enc = repeat(self.position_encoding, "... -> b ...", b=b)

        x = rearrange(x, "b ... c -> b (...) c")

        return torch.cat([x, x_enc], dim=-1)


class CosineInputAdapter(InputAdapter):
    pass