
## Data format

- A corpus file can have many dialogs
- An empty line (new line `\n`) separates dialogs.
- Each dialog can have many utterances ( an utterance is a turn)
- Each turn can have three fields, separated by tab character `\t`
  - first column is unique identifier for turn
  - second column is character or agent name
  - third column is a text of utterance

Here is an example of two dialogs
```tsv
0101_002	monica	there 's nothing to tell he 's just some guy i work with
0101_003	joey	c 'mon , you 're going out with the guy there 's gotta be something wrong with him
0101_004	chandler	all right joey , be nice . so does he have a hump ? a hump and a hairpiece ?
0101_005	phoebe	wait , does he eat chalk ?

0101_007	phoebe	just , cause , i don 't want her to go through what i went through with carl- oh
0101_008	monica	okay , everybody relax . this is not even a date . it 's just two people going out to dinner and- not having sex .
0101_009	chandler	sounds like a date to me .

```

## Some tools:

``` bash
head -100 data/friends/friends.train.tok.tsv | \
    python -m virtchar.other.friends_prep  | \
    python -m virtchar.other.simplify  --no-brackets --no-puncts --no-case -m 80


# Used for seperating the dialogs based on `<scene>` and `<event>` tags
$ python -m virtchar.other.friends_prep


# Used for text simplification
$ python -m virtchar.other.simplify -h
usage: simplify.py [-h] [-i INP] [-o OUT] [--no-case] [--no-puncts]
                   [--no-brackets] [-m MAX_LEN]

optional arguments:
  -h, --help            show this help message and exit
  -i INP, --inp INP     Input file path (default: <_io.TextIOWrapper
                        name='<stdin>' mode='r' encoding='UTF-8'>)
  -o OUT, --out OUT     Output file path (default: <_io.TextIOWrapper
                        name='<stdout>' mode='w' encoding='UTF-8'>)
  --no-case             Casing of text (default: False)
  --no-puncts           Punctuations in text. (default: False)
  --no-brackets         Brackets and the text inside the brackets. (default:
                        False)
  -m MAX_LEN, --max-len MAX_LEN
                        Maximum number of words in sequence (truncate longer
                        seqs) (default: 60)
```
