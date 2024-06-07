from arithmetic.base import ArithmeticCoderBase
from arithmetic.constants import NUM_STATE_BITS
from collections import deque
import numpy as np


class Decoder(ArithmeticCoderBase):
    def __init__(self, encoded_data: list):
        super().__init__()
        self.input = deque(encoded_data)
        self.code = sum(
            self.read_code_bit() << i for i in range(NUM_STATE_BITS - 1, -1, -1)
        )

    def decode_symbol(self, cum_freqs):
        total = int(cum_freqs[-1])

        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range

        symbol = np.searchsorted(cum_freqs, value, side="right")

        self.update(cum_freqs, symbol)

        return symbol

    def shift(self):
        self.code = ((self.code << 1) & self.state_mask) | self.read_code_bit()

    def underflow(self):
        self.code = (
            (self.code & self.half_range)
            | ((self.code << 1) & (self.state_mask >> 1))
            | self.read_code_bit()
        )

    def read_code_bit(self):
        return self.input.popleft() if len(self.input) else 0
