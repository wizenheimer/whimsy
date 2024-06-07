from arithmetic.base import ArithmeticCoderBase
from arithmetic.constants import NUM_STATE_BITS

class Encoder(ArithmeticCoderBase):
    def __init__(self):
        super().__init__()
        self.encoded_data = []
        self.num_underflow = 0

    def get_encoded(self):
        return self.encoded_data

    def encode_symbol(self, cum_freqs, symbol):
        self.update(cum_freqs, symbol)

    def finish(self):
        self.encoded_data.append(1)

    def shift(self):
        bit = self.low >> (NUM_STATE_BITS - 1)
        self.encoded_data.append(bit)
        self.encoded_data.extend([bit ^ 1] * self.num_underflow)
        self.num_underflow = 0

    def underflow(self):
        self.num_underflow += 1
