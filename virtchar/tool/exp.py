import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Iterator, List, Tuple, Union, Any
import torch
import random

from virtchar import log, load_conf
from virtchar.tool.dataprep import (RawRecord, DialogRecord, Field, LookupField,
                                    BatchIterable, LoopingIterable)
from virtchar.utils import IO
from itertools import zip_longest


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

        self.text_field = Field(str(self._text_field_file))\
            if self._text_field_file.exists() else None
        self.char_field = LookupField(str(self._char_field_file)) \
            if self._char_field_file.exists() else None

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

    def read_raw_data(self, dialog_path: Union[str, Path], max_seq_len: int,
                      text_tokenizer, char_id_mapper) \
            -> Iterator[DialogRecord]:
        recs = self.read_raw_lines(dialog_path)
        recs = ((char_id_mapper(char), text_tokenizer(dialog)) for char, dialog in recs)
        recs = ((char, dialog[:max_seq_len]) for char, dialog in recs)
        return recs

    def pre_process_parallel(self, args: Dict[str, Any]):

        # character names vocabulary
        assert 'characters' in args and type(args['characters']) is list
        self.write_lines(self._char_field_file, args['characters'])
        self.char_field = LookupField(self._char_field_file)

        # Dialog Text vocabulary
        files = [args['vocab_text']]
        no_split_toks = args.get('no_split_toks')
        self.text_field = Field.train(args['pieces'], args['max_types'],
                                      str(self._text_field_file), files,
                                      no_split_toks=no_split_toks)

        # create Piece IDs
        train_recs = self.read_raw_data(args['train_dialogs'], args['max_seq_len'],
                                        text_tokenizer=self.text_field.encode_as_ids,
                                        char_id_mapper=self.char_field.encode_as_id)
        self.write_tsv(train_recs, self.train_file)
        val_recs = self.read_raw_data(args['valid_dialogs'], args['max_seq_len'],
                                      text_tokenizer=self.text_field.encode_as_ids,
                                      char_id_mapper=self.char_field.encode_as_id)
        self.write_tsv(val_recs, self.valid_file)

        if args.get('text_files'):
            # Redo again as plain text files
            train_recs = self.read_raw_data(args['train_dialogs'], args['max_seq_len'],
                                            text_tokenizer=self.text_field.tokenize,
                                            char_id_mapper=self.char_field.remap)
            self.write_tsv(train_recs, str(self.train_file).replace('.tsv', '.pieces.tsv'))
            val_recs = self.read_raw_data(args['valid_dialogs'], args['max_seq_len'],
                                          text_tokenizer=self.text_field.tokenize,
                                          char_id_mapper=self.char_field.remap)
            self.write_tsv(val_recs, str(self.valid_file).replace('.tsv', '.pieces.tsv'))

        if args.get("finetune_src") or args.get("finetune_tgt"):
            self.pre_process_finetune(args)

        # get samples from validation set
        n_samples = args.get('num_samples', 5)

        val_raw_recs = self.read_raw_data(args['valid_dialogs'],
                                          args['max_seq_len'],
                                          text_tokenizer=lambda line: line.strip().split(),
                                          char_id_mapper=lambda line: line.strip())
        val_raw_recs = list(val_raw_recs)
        random.shuffle(val_raw_recs)
        samples = val_raw_recs[:n_samples]
        self.write_tsv(samples, self.samples_file)

    def pre_process_finetune(self, args=None):
        """
        Pre process records for fine tuning
        :param args:
        :return:
        """
        log.info("Going to prep fine tune files")
        args = args if args else self.config['prep']
        assert 'finetune_dialogs' in args
        # create Piece IDs
        finetune_recs = self.read_raw_data(args['finetune_dialogs'],
                                           args['max_seq_len'],
                                           text_tokenizer=self.text_field.encode_as_ids,
                                           char_id_mapper=self.char_field.encode_as_id)
        self.write_tsv(finetune_recs, self.finetune_file)

        if args.get('text_files'):
            # Redo again as plain text files
            finetune_recs = self.read_raw_data(args['finetune_dialogs'],
                                               args['max_seq_len'],
                                               text_tokenizer=self.text_field.tokenize,
                                               char_id_mapper=self.char_field.remap)
            self.write_tsv(finetune_recs, str(self.finetune_file).replace('.tsv', '.pieces.tsv'))

    def pre_process(self, args=None):
        args = args if args else self.config['prep']
        self.pre_process_parallel(args)

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
        log.info(f"Saving epoch {step} to {path}")
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

    def get_train_data(self, batch_size: int, steps: int = 0, sort_dec=True, batch_first=True,
                       shuffle=False, copy_xy=False, fine_tune=False):
        inp_file = self.train_file
        if fine_tune:
            if not self.finetune_file.exists():
                # user may have added fine tune file later
                self.pre_process_finetune()
            log.info("Using Fine tuning corpus instead of training corpus")
            inp_file = self.finetune_file

        train_data = BatchIterable(inp_file, batch_size=batch_size, sort_dec=sort_dec,
                                   batch_first=batch_first, shuffle=shuffle, copy_xy=copy_xy)
        if steps > 0:
            train_data = LoopingIterable(train_data, steps)
        return train_data

    def get_val_data(self, batch_size: int, sort_dec=True, batch_first=True,
                     shuffle=False, copy_xy=False):
        return BatchIterable(self.valid_file, batch_size=batch_size, sort_dec=sort_dec,
                             batch_first=batch_first, shuffle=shuffle, copy_xy=copy_xy)
