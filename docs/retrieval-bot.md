# Retrieval bot using pretrained sentence embeddings

Using https://github.com/facebookresearch/InferSentpre trained models




## 1. Prepare a sentence encoder
### a. Download

```bash
mkdir pretrained && cd pretrained

# for Infer sent v2 model (based on fasttext)
curl -Lo infersent2.pkl https://s3.amazonaws.com/senteval/infersent/infersent2.pkl

# fast text word vecs
curl -Lo crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
cd .. # go back to root dir
```


### b. Get all the training text for vocabulary
```bash
dir=data/merged  # point to data dir
cut -f3 $dir/friends.train.tok.simpl.tsv > $dir/friends.train.tok.simpl.txt
```

### c. prepare a sentence encoder model

```bash
# Add the virchar and InferSent git repo to PYTHONPATH
export PYTHONPATH=.:InferSent

python -m virtchar.retrieval.unisenc prep \
 -i pretrained/infersent2.pkl \
 -v pretrained/crawl-300d-2M.vec \
 -o botmodels/infersent2.updated.pkl \
 -s data/friends/friends.train.tok.txt
```

## 2. Filter message-response pair for characters

```bash
idir=data/merged
odir=data/chars
for ch in chandler monica joey ross rachel phoebe; do
  for sp in train test dev; do
    echo $ch $sp
    python -m virtchar.tool.filter_pairs $ch -i $idir/friends.$sp.tok.simpl.tsv \
           -o $odir/$ch.$sp.msg-resp.tok.simpl.tsv
  done
done
```

## 3. Train Chat Bot Models

```
bot=chandler
python -m virtchar.retrieval.bot prep \
 -ep botmodels/infersent2.updated.pkl -mp botmodels/$bot.bot.pkl \
 -c data/chars/$bot.train.msg-resp.tok.simpl.tsv
```

## 4. Chat

```bash
bot=chandler
python -m virtchar.retrieval.bot chat -mp botmodels/$bot.bot.pkl
```
