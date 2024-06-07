# <p align="center">Whimsy</p>  
    
LLMs like GPT (Generative Pre-trained Transformer) are trained on vast amounts of text data for predicting the next token in a sequence, making them highly effective at understanding language context and nuances. In whimsy, an LLM is used as a probabilistic model to predict the likelihood of subsequent tokens in the text. This predictive capability is crucial for efficient compression as it allows whimsy to assign shorter codes to more probable tokens, thus reducing the overall size of the encoded data.

## Arithmetic Coding

Arithmetic coding is a form of entropy encoding used in lossless data compression. Unlike traditional binary encoding, which assigns fixed-length bits to input characters, arithmetic coding encodes the entire message into a single number, a fraction n, where 0.0 ≤ n < 1.0. The efficiency of arithmetic coding comes from its ability to use fractional bits to encode symbols according to their probabilities:

- **Probabilistic Model:** The probabilities used in arithmetic coding must be as accurate as possible to achieve optimal compression. This is where the LLM comes in, providing a probability distribution for each token based on its context in the text.
- **Encoding Process:** The encoder works by narrowing down an interval based on the probabilities of each token appearing next in the sequence. This interval is continually refined as each token is processed.
- **Decoding Process:** To decode, the process is reversed. Starting with the encoded number, the decoder uses the same probabilistic model to determine which token corresponds to the current interval.

## Integration of LLM with Arithmetic Coding

The integration of an LLM with arithmetic coding in whimsy allows for the dynamic adjustment of the probability model as each token is read:

- **Dynamic Probability Adjustment:** As the LLM reads each token, it adjusts the probability model based on what it has processed, thus continually refining the compression efficiency based on the actual data.
- **Context Sensitivity:** The context window of the LLM plays a significant role. It determines how much of the previous text the model can consider when predicting the next token, which directly impacts the effectiveness of the compression.

## How It Works

### Compression

whimsy’s compression mechanism is built around the integration of Large Language Models (LLMs) with arithmetic coding, specifically tailored for handling natural language text. Here’s a deeper theoretical insight into the process:

1. **Tokenization and Model Initialization:**
   - **Tokenization:** The input text is first tokenized. This involves converting the string of text into a sequence of tokens, which are smaller, manageable units (like words or subwords) that the LLM can process.
   - **Model Initialization:** The compressor is initialized with a pre-trained LLM, which is configured to predict the probability of each token based on the sequence that precedes it.
2. **Logit to Probability Conversion:**
   - **Logits:** Logits are raw predictions from the LLM that represent the unnormalized log probabilities of each potential next token.
   - **Probability Calculation:** The logits are transformed into probabilities using a softmax function, which exponentiates and normalizes the logits, making them sum to one.
3. **Frequency and Cumulative Distribution Function (CDF):**
   - **Frequency Scaling:** Probabilities are then scaled by a constant factor (FREQ_SCALE_FACTOR) and rounded to the nearest integer to produce frequency counts for each token. These frequencies represent how often each token is expected to appear, based on the model’s predictions.
   - **Cumulative Frequencies:** The cumulative frequency distribution is calculated by summing these frequencies. This CDF is crucial for the next step—arithmetic coding.
4. **Arithmetic Coding:**
   - **Encoding:** The compressor uses the CDF to encode each token into a compact binary format. The encoding process involves narrowing down an interval based on the CDF values for each token, effectively compressing the data into fewer bits.

### Decompression

The decompression process essentially reverses the steps of compression, using the same probabilistic models to reconstruct the original text from the encoded data:

1. **Bitstream to BASE64 Conversion:**
   - The compressed data, encoded as a bitstream, is first converted from BASE64 encoding back to a binary format.
2. **Arithmetic Decoding:**
   - **Initialization:** Similar to compression, the decompressor initializes the same LLM to recreate the probability distributions for each token.
   - **Decoding Process:** Using the arithmetic coding technique, the decoder interprets the binary data. It reconstructs the original token sequence by selecting tokens that correspond to the specific intervals in the cumulative frequency distribution.
3. **Token Reconstruction:**
   - As tokens are decoded, they are collected and concatenated to reconstruct the original sequence of text.
4. **Detokenization:**
   - Finally, the sequence of tokens is converted back into a string of text, producing the original message.

### Batch Handling and Error Correction

To manage larger texts that exceed the LLM’s context window, the text is divided into manageable chunks that fit within the model’s limitations. Each chunk is processed independently, ensuring that the entire text can be efficiently compressed and decompressed without loss of context or meaning.

Error handling mechanisms are also integrated to address potential issues during tokenization or encoding, ensuring the integrity and accuracy of both compressed and decompressed data.

## Limitations and Considerations

While powerful, the approach used in whimsy comes with specific limitations:

   - **Inference Speed**: The speed of the LLM’s inference can significantly affect both compression and decompression speeds. This is because each token’s encoding and decoding depend on the LLM’s output, which can be computationally intensive.
   - **Context Window Size**: The maximum size of input text that can be effectively compressed in one pass is limited by the LLM’s context window. Larger texts need to be broken down or require batching mechanisms, which are currently under development.

## References      
- [Large Text Compression Benchmark](https://www.mattmahoney.net/dc/text.html)
- [Language Modelling is Compression](https://arxiv.org/pdf/2309.10668)
- [Text Compression using Large Language Models](https://bellard.org/ts_zip/)
- [Compression Through Language Modeling](https://nlp.stanford.edu/courses/cs224n/2006/fp/aeldaher-jconnor-1-report.pdf)
- [llama-zip](https://github.com/AlexBuz/llama-zip) 👑
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
