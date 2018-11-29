A simple retrieval based chat bot


## 1. Prepare a sentence encoder

### a. Get all the training text for vocabulary
```bash
dir=data/merged
cut -f3 $dir/friends.train.tok.simpl.tsv > $dir/friends.train.tok.simpl.txt
```

### b. prepare a sentence encoder model

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
for ch in chandler monica joey ross rachel phoebe
do
   for sp in train test dev
   do
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
python -m virtchar.retrieval.bot prep -mp
```
