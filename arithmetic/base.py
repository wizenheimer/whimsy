from arithmetic.constants import NUM_STATE_BITS

class ArithmeticCoderBase:
    def __init__(self):
        full_range = 1 << NUM_STATE_BITS
        self.half_range = full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.state_mask = full_range - 1
        self.low = 0
        self.high = self.state_mask

    def update(self, cum_freqs, symbol):
        total = int(cum_freqs[-1])

        range = self.high - self.low + 1

        symhigh = int(cum_freqs[symbol])
        self.high = self.low + symhigh * range // total - 1

        symlow = int(cum_freqs[symbol - 1]) if symbol > 0 else 0
        self.low = self.low + symlow * range // total

        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = (self.low << 1) & self.state_mask
            self.high = ((self.high << 1) & self.state_mask) | 1

        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()
