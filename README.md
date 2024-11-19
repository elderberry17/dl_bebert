# BeBert

The Be(-zgin)Bert is the silliest implementation of the BERT architecture with a full and detailed code of basic BERT-like blocks from scratch (using only the most simple torch layers).

There are implementations of FeedForward, NormLayer, Semantic and Position embeddings (position embeddings here are absolute, based on trigonometric functions).

# Train

The repository also include data preparation and training steps for NSP and MLM tasks. The NSP task here though is represented as classification if the pair is the same sentence in 2 different languages (Russian and English) to force the model become a multilingual at the pretraining step. The used data is Yandex Translate Corpus (https://translate.yandex.ru/corpus)

The tokenizer is stolen from here (https://huggingface.co/cointegrated/rubert-tiny). However, it is a great idea to add code from training your own tokenizer from scratch!

You are free to change data/configs/traning process to make the repository a better place for students.

# Test

The test stage is using the MTEB (https://github.com/embeddings-benchmark/mteb) corpus of Benchmarks. There is a test config file that by default employs only a list of benchmarks for Russian language.

# HF model

There is a ready-to-go trained model on Hugging-Face (https://huggingface.co/elderberry17/silly-bert).

The model was trained on this corpus during 1 epoch with MLM (with dynamic masking as RoBERTa does) and then 1 epoch with the decribed above NSP task.

The training process has taken approxiamately 8 hours with batch size 8 for MLM and 128 for NSP. The used card is NVIDIA-A800. 

# Results

You are free to check the results on MTEB in the corresponding folder.