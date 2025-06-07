# 🧠 Text Prediction Using LSTM

This repository contains a deep learning project that uses an LSTM (Long Short-Term Memory) neural network to perform next-word prediction based on a given text corpus. The model is trained on the writings of Benjamin Franklin and can generate text sequences when given a seed phrase.

---

## 📌 Objective

The primary objective of this project is to:

- Learn patterns and semantics from historical English text.
- Predict the most likely next word given a sequence of previous words.
- Generate new, coherent text based on a user-provided input phrase.

This project serves as an educational example of natural language processing (NLP), sequence modeling, and deep learning for text generation using TensorFlow and Keras.

---

## 🛠️ Model Architecture

The model is a simple Sequential neural network built with Keras. It consists of:

- **Embedding Layer**: Converts words into dense vector representations.
- **LSTM Layer**: Learns dependencies between sequential words.
- **Dense Output Layer**: Applies a softmax activation to predict the next word.

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_len-1))
model.add(LSTM(units=100))
model.add(Dense(units=vocab_size, activation='softmax'))
```
## 📚 Dataset Details

- **File**: `franklin.txt`
- **Content**: Likely consists of public domain writings or speeches by Benjamin Franklin.

### 🧼 Preprocessing Steps:
- Removal of newlines, special characters (like quotation marks), and duplicate spaces.
- Tokenization using Keras `Tokenizer` (at the word level).
- Generation of input-output sequences for supervised training.
- Categorical one-hot encoding of output labels to be used in classification.

---

## 🔁 Sequence Generation Process

The model is trained on short sequences of words, where the objective is to predict the next word given a sequence of prior words. A sliding window technique is used to create these input-output pairs.

### 🔤 Example:
Given the sentence:
"education is the most powerful weapon"


We generate the following training pair:
- **Input**: `"education is the"`
- **Output**: `"most"`

The process continues across the entire corpus, generating all possible sub-sequences. Each input sequence is one word shorter than the full sequence, and the output is the next immediate word.

This type of training helps the LSTM learn the likelihood of a word following a given context.

