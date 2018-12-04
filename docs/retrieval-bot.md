# Retrieval bot using pretrained sentence embeddings

Using https://github.com/facebookresearch/InferSent pre-trained models



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
mkdir data/retrieval
cut -f3 data/friends/friends.train.raw.tsv > data/retrieval/friends.train.raw.txt
```

### c. prepare a sentence encoder model

```bash
# InferSent is a submodule
git submodule init

# Add the virchar and InferSent git repo to PYTHONPATH
export PYTHONPATH=$PWD:$PWD/InferSent

python -m virtchar.retrieval.unisenc prep \
 -i pretrained/infersent2.pkl \
 -v pretrained/crawl-300d-2M.vec \
 -o botmodels/infersent2.wvec.pkl \
 -s data/retrieval/friends.train.raw.txt
```

## 2. Filter message-response pair for characters

```bash
idir=data/friends
odir=data/chars
for ch in Chandler Monica Joey Ross Rachel Phoebe; do
  for sp in train test dev; do
    echo $ch $sp
    python -m virtchar.tool.filter_pairs $ch -i $idir/friends.$sp.raw.tsv \
           -o $odir/$ch.$sp.msg-resp.raw.tsv
  done
done
```

## 3. Train Chat Bot Models

```
bot=Chandler
python -m virtchar.retrieval.bot prep \
 -ep botmodels/infersent2.wvec.pkl -mp botmodels/$bot.bot.pkl \
 -c data/chars/$bot.train.msg-resp.raw.tsv
```

Make all other bots too, make sure to run this on GPU, because it takes lot of time on CPU.

```
for bot in Monica Ross Rachel Joey Phoebe; do
  echo "Making $bot";
  python -m virtchar.retrieval.bot prep \
         -ep botmodels/infersent2.wvec.pkl \
         -mp botmodels/$bot.bot.pkl \
         -c data/chars/$bot.train.msg-resp.raw.tsv
done
```

## 4. Chat

```bash
bot=Chandler
python -m virtchar.retrieval.bot chat -mp botmodels/$bot.bot.pkl
```


