# 🧠 Text Prediction Using LSTM

This project implements a word-level text prediction model using an LSTM-based neural network in TensorFlow/Keras. The model is trained on a text corpus and is capable of predicting the next word(s) given a seed input.

---

## 📌 Objective

The primary goal of this project is to:
- Learn the language structure from a given corpus (e.g., `franklin.txt`)
- Predict the next word based on a sequence of previous words
- Generate coherent and meaningful text sequences

---

## 🛠️ Model Architecture

The model is built using TensorFlow's Keras API with the following architecture:

```python
Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(units=100),
    Dense(units=vocab_size, activation='softmax')
])
