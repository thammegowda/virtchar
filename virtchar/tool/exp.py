import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Iterator, List, Tuple, Union, Any, Set
import torch
import random
from collections import Counter

from virtchar import log, load_conf
from virtchar.tool.dataprep import (
    RawRecord, DialogRecord, Field, LookupField, RawDialogReader, Dialog,
    DialogReader, DialogBatchReader, DialogMiniBatch, LoopingIterable)
from virtchar.utils import IO


class DialogExperiment:

    def __init__(self, work_dir: Union[str, Path], read_only=False,
                 config: Optional[Dict[str, Any]] = None):
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        log.info(f"Initializing an experiment. Directory = {work_dir}")
        self.read_only = read_only
        self.work_dir = work_dir
        self.data_dir = work_dir / 'data'
        self.model_dir = work_dir / 'models'
        self._config_file = work_dir / 'conf.yml'
        self._text_field_file = self.data_dir / 'text.model'
        self._char_field_file = self.data_dir / 'vocab.char.txt'
        self._prepared_flag = self.work_dir / '_PREPARED'
        self._trained_flag = self.work_dir / '_TRAINED'

        self.train_file = self.data_dir / 'train.tsv.gz'
        self.finetune_file = self.data_dir / 'finetune.tsv.gz'
        self.valid_file = self.data_dir / 'valid.tsv.gz'
        # a set of samples to watch the progress qualitatively
        self.samples_file = self.data_dir / 'samples.tsv.gz'

        if not read_only:
            for _dir in [self.model_dir, self.data_dir]:
                if not _dir.exists():
                    _dir.mkdir(parents=True)
        if type(config) is str:
            config = load_conf(config)
        self.config = config if config else load_conf(self._config_file)

        self.text_field = Field(str(self._text_field_file)) \
            if self._text_field_file.exists() else None
        self.char_field = LookupField(str(self._char_field_file)) \
            if self._char_field_file.exists() else None

        # these are the characters to which we optimize the loss
        self._model_chars = None

    def store_config(self):
        with IO.writer(self._config_file) as fp:
            return yaml.dump(self.config, fp, default_flow_style=False)

    @property
    def model_type(self) -> Optional[str]:
        return self.config.get('model_type')

    @model_type.setter
    def model_type(self, mod_type: str):
        self.config['model_type'] = mod_type

    def has_prepared(self):
        return self._prepared_flag.exists()

    def has_trained(self):
        return self._trained_flag.exists()

    @staticmethod
    def write_tsv(records: Iterator[DialogRecord], path: Union[str, Path]):
        seqs = ((str(x), ' '.join(map(str, y))) for x, y in records)
        lines = (f'{x}\t{y}\n' for x, y in seqs)
        log.info(f"Storing data at {path}")
        with IO.writer(path) as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def read_raw_lines(dialog_path: Union[str, Path]) -> Iterator[RawRecord]:
        with IO.reader(dialog_path) as lines:
            recs = (line.split("\t")[-2:] for line in lines)
            recs = ((char.strip(), dialog.strip()) for char, dialog in recs)
            recs = ((char, dialog) for char, dialog in recs if char and dialog)
            yield from recs

    @staticmethod
    def write_lines(path: Union[str, Path], lines):
        count = 0
        with IO.writer(path) as out:
            for line in lines:
                count += 1
                out.write(line.strip())
                out.write("\n")
            log.info(f"Wrote {count} lines to {path}")

    @staticmethod
    def scan_characters(path: Path, min_freq: int, unk='<unk>', pad='<pad>') -> List[str]:
        def _read_char_names():
            with IO.reader(path) as inp:
                for line in inp:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        yield parts[-2]

        stats = Counter(_read_char_names())

        stats = [(char_name, count) for char_name, count in stats.items() if count >= min_freq]
        stats = sorted(stats, key=lambda pair: pair[1], reverse=True)
        char_names = [cn for cn, ct in stats]
        placeholder = [tok for tok in [pad, unk] if tok]
        char_names = placeholder + char_names
        return char_names

    def pre_process_train_dev(self, args: Dict[str, Any]):

        # character names vocabulary
        if self.char_field and self._char_field_file.exists():
            log.warning("Skipping character vocab creating. since it already exists")
            self.char_field = LookupField(self._char_field_file)
        else:
            char_min_freq = args.get('char_min_freq', 500)
            log.info(f"Scanning characters in training data with with freq {char_min_freq}")
            char_names = self.scan_characters(args['train_dialogs'], min_freq=char_min_freq)
            log.info(f"Found {len(char_names)} characters")
            self.write_lines(self._char_field_file, char_names)
            self.char_field = LookupField(self._char_field_file)

        # Dialog Text vocabulary
        if self._text_field_file.exists() and self.text_field is not None:
            log.warning("Skipping the vocab creation since it already exist")
            self.text_field = Field(self._text_field_file)
        else:
            files = [args['vocab_text']]
            no_split_toks = args.get('no_split_toks')
            self.text_field = Field.train(args['pieces'], args['max_types'],
                                          str(self._text_field_file), files,
                                          no_split_toks=no_split_toks)

        # create Piece IDs
        for key, out_path in [('train_dialogs', self.train_file),
                              ('valid_dialogs', self.valid_file)]:
            dialogs = RawDialogReader(args[key],
                                      text_field=self.text_field,
                                      char_field=self.char_field,
                                      max_seq_len=args['max_seq_len'])
            self.write_dialogs(dialogs, out_path)

        if args.get("finetune_src") or args.get("finetune_tgt"):
            self.pre_process_finetune(args)

        # get samples from validation set
        n_samples = args.get('num_samples', 5)
        samples = self.pick_samples(Path(args['valid_dialogs']), n_samples)
        self.write_dialogs(samples, self.samples_file)

    @staticmethod
    def pick_samples(path: Path, n_samples: int, min_len: int = 6, max_len: int = 15):
        dialogs = list(RawDialogReader(path))
        dialogs = [d for d in dialogs if min_len <= len(d) <= max_len]
        random.shuffle(dialogs)
        return dialogs[:n_samples]

    @staticmethod
    def write_dialogs(dialogs: Iterator[Dialog], out: Path, dialog_sep='\n'):
        count = 0
        with IO.writer(out) as outh:
            for dialog in dialogs:
                count += 1
                for utter in dialog.chat:
                    if utter.uid:
                        outh.write(f'{utter.uid}\t')
                    text = " ".join(map(str, utter.text))
                    outh.write(f'{utter.char}\t{text}\n')
                outh.write(dialog_sep)
        log.info(f"Wrote {count} recs to {out}")

    def pre_process_finetune(self, args=None):
        """
        Pre process records for fine tuning
        :param args:
        :return:
        """
        log.info("Going to prep fine tune files")
        args = args if args else self.config['prep']
        assert 'finetune_dialogs' in args
        dialogs = RawDialogReader(args['finetune_dialogs'],
                                  text_field=self.text_field,
                                  char_field=self.char_field,
                                  max_seq_len=args['max_seq_len'])
        self.write_dialogs(dialogs, self.finetune_file)

    def pre_process(self, args=None):
        args = args if args else self.config['prep']
        self.pre_process_train_dev(args)

        # update state on disk
        self.persist_state()
        self._prepared_flag.touch()

    def persist_state(self):
        """Writes state of current experiment to the disk"""
        assert not self.read_only
        if 'model_args' not in self.config:
            self.config['model_args'] = {}
        args = self.config['model_args']
        args['text_vocab'] = len(self.text_field) if self.text_field else 0
        args['char_vocab'] = len(self.char_field) if self.char_field else 0
        self.config['updated_at'] = datetime.now().isoformat()
        self.store_config()

    def store_model(self, step: int, model, train_score: float, val_score: float, keep: int):
        """
        saves model to a given path
        :param step: step number of training
        :param model: model object itself
        :param train_score: score of model on training split
        :param val_score: score of model on validation split
        :param keep: number of good models to keep, bad models will be deleted
        :return:
        """
        # TODO: improve this by skipping the model save if the model is not good enough to be saved
        if self.read_only:
            log.warning("Ignoring the store request; experiment is readonly")
            return
        name = f'model_{step:05d}_{train_score:.6f}_{val_score:.6f}.pkl'
        path = self.model_dir / name
        log.info(f"Saving... step={step} to {path}")
        torch.save(model, str(path))

        for bad_model in self.list_models(sort='total_score', desc=False)[keep:]:
            log.info(f"Deleting bad model {bad_model} . Keep={keep}")
            os.remove(str(bad_model))

        with IO.writer(os.path.join(self.model_dir, 'scores.tsv'), append=True) as f:
            cols = [str(step), datetime.now().isoformat(), name, f'{train_score:g}',
                    f'{val_score:g}']
            f.write('\t'.join(cols) + '\n')

    @staticmethod
    def _path_to_validn_score(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        valid_score = float(parts[-1])
        return valid_score

    @staticmethod
    def _path_to_total_score(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        tot_score = float(parts[-2]) + float(parts[-1])
        return tot_score

    @staticmethod
    def _path_to_step_no(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        step_no = int(parts[-3])
        return step_no

    def list_models(self, sort: str = 'step', desc: bool = True) -> List[Path]:
        """
        Lists models in descending order of modification time
        :param sort: how to sort models ?
          - valid_score: sort based on score on validation set
          - total_score: sort based on validation_score + training_score
          - mtime: sort by modification time
          - step (default): sort by step number
        :param desc: True to sort in reverse (default); False to sort in ascending
        :return: list of model paths
        """
        paths = self.model_dir.glob('model_*.pkl')
        sorters = {
            'valid_score': self._path_to_validn_score,
            'total_score': self._path_to_total_score,
            'mtime': lambda p: p.stat().st_mtime,
            'step': self._path_to_step_no
        }
        if sort not in sorters:
            raise Exception(f'Sort {sort} not supported. valid options: {sorters.keys()}')
        return sorted(paths, key=sorters[sort], reverse=desc)

    def _get_first_model(self, sort: str, desc: bool) -> Tuple[Optional[Path], int]:
        """
        Gets the first model that matches the given sort criteria
        :param sort: sort mechanism
        :param desc: True for descending, False for ascending
        :return: Tuple[Optional[Path], step_num:int]
        """
        models = self.list_models(sort=sort, desc=desc)
        if models:
            _, step, train_score, valid_score = models[0].name.replace('.pkl', '').split('_')
            return models[0], int(step)
        else:
            return None, -1

    def get_best_known_model(self) -> Tuple[Optional[Path], int]:
        """Gets best Known model (best on lowest scores on training and validation sets)
        """
        return self._get_first_model(sort='total_score', desc=False)

    def get_last_saved_model(self) -> Tuple[Optional[Path], int]:
        return self._get_first_model(sort='step', desc=True)

    @property
    def model_args(self) -> Optional[Dict]:
        """
        Gets args from file
        :return: args if exists or None otherwise
        """
        return self.config.get('model_args')

    @model_args.setter
    def model_args(self, model_args):
        """
        set model args
        """
        self.config['model_args'] = model_args

    @property
    def optim_args(self) -> Tuple[Optional[str], Dict]:
        """
        Gets optimizer args from file
        :return: optimizer args if exists or None otherwise
        """
        opt_conf = self.config.get('optim')
        if opt_conf:
            return opt_conf.get('name'), opt_conf.get('args')
        else:
            return None, {}

    @optim_args.setter
    def optim_args(self, optim_args: Tuple[str, Dict]):
        """
        set optimizer args
        """
        name, args = optim_args
        self.config['optim'] = {'name': name, 'args': args}

    def get_train_data(self, shuffle=False, fine_tune=False, loop_steps=0) \
            -> Iterator[DialogMiniBatch]:
        assert not shuffle, 'Not supported at the moment'
        inp_file = self.train_file
        if fine_tune:
            if not self.finetune_file.exists():
                # user may have added fine tune file later
                self.pre_process_finetune()
            log.info("Using Fine tuning corpus instead of training corpus")
            inp_file = self.finetune_file

        reader = DialogReader(inp_file)

        train_data = DialogBatchReader(reader,
                                       min_ctx=self.min_ctx,
                                       max_ctx=self.max_ctx,
                                       max_dialogs=self.max_utters,
                                       max_utters=self.max_utters,
                                       model_chars=None,
                                       min_resp_len=self.min_resp_len)
        return LoopingIterable(train_data, total=loop_steps) if loop_steps > 0 else train_data

    def get_val_data(self) -> Iterator[DialogMiniBatch]:
        reader = DialogReader(self.valid_file)
        return DialogBatchReader(reader,
                                 min_ctx=self.min_ctx,
                                 max_ctx=self.max_ctx,
                                 max_dialogs=self.max_utters,
                                 max_utters=self.max_utters,
                                 model_chars=None)

    @property
    def model_characters(self) -> Set[int]:
        raise Exception("Unsupported Op")

    @property
    def min_ctx(self):
        return self.config['trainer']['min_ctx']

    @property
    def max_ctx(self):
        return self.config['trainer']['max_ctx']

    @property
    def max_dialogs(self):
        return self.config['trainer']['max_dialogs']

    @property
    def max_utters(self):
        return self.config['trainer']['max_utters']

    @property
    def min_resp_len(self):
        return self.config.get('trainer', {}).get('min_resp_len', -1)
