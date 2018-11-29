#!/usr/bin/env bash

# for Infer sent v2 model (based on fasttext)
curl -Lo infersent2.pkl https://s3.amazonaws.com/senteval/infersent/infersent2.pkl


# fast text word vecs
curl -Lo crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip

