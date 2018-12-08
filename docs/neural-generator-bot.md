# Generator Bot

this is the page for knowing how to use the generator based dialog bot.
Here we have Hierarchical Transformer and Hierarchical RNN model.
 (Beware: RNN model was not tested after recent changes, )

# Terminology
## For Operator
- Experiment : An experiment is a top level entity, it includes model, trainer, optimizer,
dataset, config etc
- Config : An yaml file where you control the experiment. This is the only file you need to torch

## Inside the code
- Dialog  : A dialog is a sequence of Utterances naturally occurring somewhere  (like a TV show).
We dont control how long the sequences are and how many turns.
- Chat : A chat is minified dialog, we control the lengths of dialog for the practical reasons such
as fitting them into memory without ever crashing OOM.
- Utterance: An utterance is a tuple of Speaker aka Character and text
- Character: A character is a TV character (dont confuse it with the characters in string).
- Text : the sequence of words.
- Context: It means same as the Chat, except the last Utterance.
- Response: The last utterance of Chat (or the one that immediately follows context)

### Tools


* `python -m virtchar.prep` - for preparing an experiment
* `python -m virtchar.train` - for training a model
* `python -m virtchar.decode -it -sc ` - for decoding

You should be using them in the same order : prepare -> train -> decode

Add the root directory of this repository to PYTHONPATH by `export PYTHONPATH=<path here>`


## Config

The crucial part is the config. You have to Learn by examples 🙃
See that there are two examples in `example-confs` directory:
1. example-confs/hiero-transformer-charmemb.yml -- the one for hierarchical transformer
2. example-confs/rnn.conf.yml -- the one for hierarchical RNN (edit rnn_type=GRU or LSTM)


## Data format:

here is an example:
```tsv
0101_002	monica	there 's nothing to tell he 's just some guy i work with
0101_003	joey	c 'mon , you 're going out with the guy there 's gotta be something wrong with him
0101_004	chandler	all right joey , be nice . so does he have a hump ? a hump and a hairpiece ?
0101_005	phoebe	wait , does he eat chalk ?

0101_007	phoebe	just , cause , i don 't want her to go through what i went through with carl- oh
0101_008	monica	okay , everybody relax . this is not even a date . it 's just two people going out to dinner and- not having sex .
0101_009	chandler	sounds like a date to me .

```
An empty line separates two dialogs.  Each dialog has many utterances.
Each utterance has  three fields: Id, SpeakerName, Text

see `docs/dataprep.md` for tools related to data prep


## Step by Step Guide

### Step 0: Prerequisites:
1. All the necessary libs are installed (see requirements.txt)
2. Data is nicely formatted
3. You have created a config file (or used the example and edited the data paths)

Add the root of this directory tot `export PYTHONPATH=<path>`


### Step 1: Prepare an experiment

```bash
$ python -m virtchar.prep -h
usage: virtchar.prep [-h] work_dir conf_file

prepare NMT experiment

positional arguments:
  work_dir    Working directory
  conf_file   Config File
```

Example:
```bash
$ mkdir runs
$ rtg.prep [-h] runs/001-tfm example-confs/hiero-transformer-charmemb.yml
```


### Step 2: Train a model

```
$ python -m virtchar.train -h
usage: virtchar.train [-h] [-rs SEED] [-st STEPS] [-cp CHECK_POINT]
                      [-km KEEP_MODELS] [-bs BATCH_SIZE] [-op {ADAM,SGD}]
                      [-oa OPTIM_ARGS] [-ft]
                      work_dir

Train Dialog model

positional arguments:
  work_dir              Working directory

optional arguments:
  -h, --help            show this help message and exit
  -rs SEED, --seed SEED
                        Seed for random number generator. Set it to zero to
                        not touch this part. (default: 0)
  -st STEPS, --steps STEPS
                        Total steps (default: 128000)
  -cp CHECK_POINT, --check-point CHECK_POINT
                        Store model after every --check-point steps (default:
                        1000)
  -km KEEP_MODELS, --keep-models KEEP_MODELS
                        Number of checkpoints to keep. (default: 10)
  -op {ADAM,SGD}, --optim {ADAM,SGD}
                        Name of optimizer (default: ADAM)
  -oa OPTIM_ARGS, --optim-args OPTIM_ARGS
                        Comma separated key1=val1,key2=val2 args to optimizer.
                        Example: lr=0.01,warmup_steps=1000 The arguments
                        depends on the choice of --optim (default: lr=0.001)
  -ft, --fine-tune      Use fine tune corpus instead of train corpus.
                        (default: False)
```

Usage:
```
python virtchar.train runs/001-tfm  -st 64000 -cp 1000 -km 20
# train upto 64000 optimizer steps, check point every 1000 steps, keep only the best 20 models on disk
```


### Step 3: Chat with the bot:
```
python -m virtchar.decode -h
usage: rtg.decode [-h] [-if INPUT] [-of OUTPUT] [-it] [-sc] [-en ENSEMBLE]
                  work_dir [model_path [model_path ...]]

Decode using NMT model

positional arguments:
  work_dir              Working directory
  model_path            Path to model's checkpoint. If not specified, a best
                        model (based on the score on validation set) from the
                        experiment directory will be used. If multiple paths
                        are specified, then an ensembling is performed by
                        averaging the param weights (default: None)

optional arguments:
  -h, --help            show this help message and exit
  -it, --interactive    Open interactive shell with decoder (default: False)
  -sc, --skip-check     Skip Checking whether the experiment dir is prepared
                        and trained (default: False)
  -en ENSEMBLE, --ensemble ENSEMBLE
                        Ensemble best --ensemble models by averaging them
                        (default: 1)

(ignore the other options, they aren't yet complete)
```

Example usage to get an interactive shell:

```bash
python virtchar.decode runs/001-tfm -it -sc
```
