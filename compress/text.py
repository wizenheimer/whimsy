from arithmetic.constants import BASE64, FREQ_SCALE_FACTOR
from collections import deque
from arithmetic.encoder import Encoder, Decoder
import numpy as np
from more_itertools import consume
import tqdm


class TextCompressor:
    def __init__(self, model, pattern=BASE64):
        self.model = model
        self.BASE64 = pattern

    def compute_cdf(self, logits):
        logprobs = self.model.logits_to_logprobs(logits)
        probs = np.exp(logprobs)
        freqs = np.maximum(1, np.round(FREQ_SCALE_FACTOR * probs))
        cum_freqs = np.cumsum(freqs)
        return cum_freqs

    def compress(self, string):
        def bits_to_base64(bits):
            while bits and bits[-1] == 0:
                bits.pop()
            while len(bits) % 6 != 0:
                bits.append(0)
            return "".join(
                self.BASE64[int("".join(str(bit) for bit in bits[i : i + 6]), 2)]
                for i in range(0, len(bits), 6)
            )

        def process_logits(_, logits):
            next_token = tokens.popleft()
            cdf = self.compute_cdf(logits)
            encoder.encode_symbol(cdf, next_token)
            logits[next_token] = np.inf
            return logits

        def should_stop(_, logits):
            return np.argmax(logits) == self.model.token_eos()

        def encode(chunk):
            consume(
                tqdm(
                    self.model.generate(
                        tokens=[self.model.token_bos()],
                        temp=0.0,
                        logits_processor=process_logits,
                        stopping_criteria=should_stop,
                    ),
                    total=len(chunk) - 1,
                    mininterval=1 / 30,
                    desc="Compressing",
                    unit="tok",
                    leave=False,
                    dynamic_ncols=True,
                )
            )
            encoder.finish()
            compressed = bits_to_base64(encoder.get_encoded())
            return compressed

        self.model.reset()
        tokens = deque(self.model.tokenize(string.encode("utf-8"), add_bos=False))
        eos_token = self.model.token_eos()
        if len(tokens) >= self.model.n_ctx():
            # attempt batching
            chunk_size = self.model.n_ctx() - 1
            chunks = [
                tokens[i : i + chunk_size] + [eos_token]
                for i in range(0, len(tokens), chunk_size)
            ]
        else:
            tokens.append(eos_token)
            chunks = [tokens]

        encoder = Encoder()

        compressed_chunks = [
            encode(chunk)
            for chunk in tqdm(
                chunks,
                desc="Compressing Batch",
                unit="chunk",
            )
        ]

        return compressed_chunks

    def decompress(self, compressed_chunks):

        def decode(compressed):
            def base64_to_bits(string):
                bits = [
                    int(bit)
                    for char in string
                    for bit in f"{self.BASE64.index(char):06b}"
                ]
                return bits

            def process_logits(_, logits):
                cdf = self.compute_cdf(logits)
                next_token = decoder.decode_symbol(cdf)
                logits[next_token] = np.inf
                if next_token == self.model.token_eos():
                    return logits
                tokens.append(next_token)
                output_buf.extend(self.model.detokenize([next_token]))
                try:
                    output_buf.decode("utf-8")
                    output_buf.clear()
                except UnicodeDecodeError:
                    pass
                return logits

            def should_stop(_, logits):
                return np.argmax(logits) == self.model.token_eos()

            self.model.reset()
            tokens = []
            encoded = base64_to_bits(compressed)
            decoder = Decoder(encoded)
            output_buf = bytearray()
            consume(
                self.model.generate(
                    tokens=[self.model.token_bos()],
                    temp=0.0,
                    logits_processor=process_logits,
                    stopping_criteria=should_stop,
                )
            )
            decompressed = self.model.detokenize(tokens).decode("utf-8")
            return decompressed

        decompressed_chunks = [
            decode(chunk)
            for chunk in tqdm(
                compressed_chunks,
                desc="Decompressing Batch",
                unit="chunk",
            )
        ]

        return decompressed_chunks
