import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Iterator, List, Tuple, Union, Any
import torch
import random

from virtchar import log, load_conf
from virtchar.tool.dataprep import (RawRecord, ParallelSeqRecord, MonoSeqRecord,
                                    Field, BatchIterable, LoopingIterable)
from virtchar.utils import IO, line_count
from itertools import zip_longest


class TranslationExperiment:

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
        self._shared_field_file = str(self.data_dir / 'sentpiece.shared.model')
        self._src_field_file = str(self.data_dir / 'sentpiece.src.model')
        self._tgt_field_file = str(self.data_dir / 'sentpiece.tgt.model')
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

        self.shared_field, self.src_field, self.tgt_field = [
            Field(f) if Path(f).exists() else None
            for f in (self._shared_field_file, self._src_field_file, self._tgt_field_file)]

        # Either shared field  OR  individual  src and tgt fields
        assert not (self.shared_field and self.src_field)
        assert not (self.shared_field and self.tgt_field)
        # both are set or both are unset
        assert (self.src_field is None) == (self.tgt_field is None)

        self._unsupervised = self.model_type in {'binmt'}
        if self._unsupervised:
            self.mono_train_src = self.data_dir / 'mono.train.src.gz'
            self.mono_train_tgt = self.data_dir / 'mono.train.tgt.gz'
            self.mono_valid_src = self.data_dir / 'mono.valid.src.gz'
            self.mono_valid_tgt = self.data_dir / 'mono.valid.tgt.gz'

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
    def write_tsv(records: Iterator[ParallelSeqRecord], path: Union[str, Path]):
        seqs = ((' '.join(map(str, x)), ' '.join(map(str, y))) for x, y in records)
        lines = (f'{x}\t{y}\n' for x, y in seqs)
        log.info(f"Storing data at {path}")
        with IO.writer(path) as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def write_mono_lines(records: Iterator[MonoSeqRecord], path: Union[str, Path]):
        lines = (' '.join(map(str, rec)) + '\n' for rec in records)
        log.info(f"Storing data at {path}")
        with IO.writer(path) as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def read_raw_lines(src_path: Union[str, Path], tgt_path: Union[str, Path]) \
            -> Iterator[RawRecord]:
        with IO.reader(src_path) as src_lines, IO.reader(tgt_path) as tgt_lines:
            # if you get an exception here --> files have un equal number of lines
            recs = ((src.strip(), tgt.strip()) for src, tgt in zip_longest(src_lines, tgt_lines))
            recs = ((src, tgt) for src, tgt in recs if src and tgt)
            yield from recs

    def read_raw_data(self, src_path: Union[str, Path], tgt_path: Union[str, Path],
                      truncate: bool, src_len: int, tgt_len: int, tokenizer) \
            -> Iterator[ParallelSeqRecord]:
        recs = self.read_raw_lines(src_path, tgt_path)
        recs = ((tokenizer(x), tokenizer(y)) for x, y in recs)
        if truncate:
            recs = ((src[:src_len], tgt[:tgt_len]) for src, tgt in recs)
        else:  # Filter out longer sentences
            recs = ((src, tgt) for src, tgt in recs if len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    @staticmethod
    def read_mono_raw_data(path: Union[str, Path], truncate: bool, max_len: int, tokenizer):
        with IO.reader(path) as lines:
            recs = (tokenizer(line.strip()) for line in lines if line.strip())
            if truncate:
                recs = (rec[:max_len] for rec in recs)
            else:  # Filter out longer sentences
                recs = (rec for rec in recs if 0 < len(rec) <= max_len)
            yield from recs

    def pre_process_parallel(self, args: Dict[str, Any]):
        assert args['shared_vocab']  # TODO support individual vocab types
        files = [args['train_src'], args['train_tgt']]
        for val in [args.get('mono_src'), args.get('mono_tgt')]:
            if val:
                files.extend(val)

        # check if files are parallel
        assert line_count(args['train_src']) == line_count(args['train_tgt'])
        assert line_count(args['valid_src']) == line_count(args['valid_tgt'])

        no_split_toks = args.get('no_split_toks')
        self.shared_field = Field.train(args['pieces'], args['max_types'],
                                        self._shared_field_file, files,
                                        no_split_toks=no_split_toks)

        # create Piece IDs
        train_recs = self.read_raw_data(args['train_src'], args['train_tgt'], args['truncate'],
                                        args['src_len'], args['tgt_len'],
                                        tokenizer=self.src_vocab.encode_as_ids)
        self.write_tsv(train_recs, self.train_file)
        val_recs = self.read_raw_data(args['valid_src'], args['valid_tgt'], args['truncate'],
                                      args['src_len'], args['tgt_len'],
                                      tokenizer=self.tgt_vocab.encode_as_ids)
        self.write_tsv(val_recs, self.valid_file)

        if args.get('text_files'):
            # Redo again as plain text files
            train_recs = self.read_raw_data(args['train_src'], args['train_tgt'], args['truncate'],
                                            args['src_len'], args['tgt_len'],
                                            tokenizer=self.src_vocab.tokenize)
            self.write_tsv(train_recs, str(self.train_file).replace('.tsv', '.pieces.tsv'))
            val_recs = self.read_raw_data(args['valid_src'], args['valid_tgt'], args['truncate'],
                                          args['src_len'], args['tgt_len'],
                                          tokenizer=self.tgt_vocab.tokenize)
            self.write_tsv(val_recs, str(self.valid_file).replace('.tsv', '.pieces.tsv'))

        if args.get("finetune_src") or args.get("finetune_tgt"):
            self.pre_process_finetune(args)

        # get samples from validation set
        n_samples = args.get('num_samples', 5)
        val_raw_recs = self.read_raw_data(args['valid_src'], args['valid_tgt'], args['truncate'],
                                          args['src_len'], args['tgt_len'],
                                          tokenizer=lambda line: line.strip().split())
        val_raw_recs = list(val_raw_recs)
        random.shuffle(val_raw_recs)
        samples = val_raw_recs[:n_samples]
        self.write_tsv(samples, self.samples_file)

    def pre_process_mono(self, args):
        no_split_toks = args.get('no_split_toks')
        if args.get('shared_vocab'):
            files = [args['mono_train_src'], args['mono_train_tgt']]
            self.shared_field = Field.train(args['pieces'],
                                            args['max_types'],
                                            self._shared_field_file, files,
                                            no_split_toks=no_split_toks)
        else:
            self.src_field = Field.train(args['pieces'], args['max_src_types'],
                                         self._src_field_file, [args['mono_train_src']],
                                         no_split_toks=no_split_toks)

            self.tgt_field = Field.train(args['pieces'], args['max_tgt_types'],
                                         self._tgt_field_file, [args['mono_train_tgt']],
                                         no_split_toks=no_split_toks)

        def _prep_file(raw_file, out_file, do_truncate, max_len, field: Field):
            recs = self.read_mono_raw_data(raw_file, do_truncate, max_len, field.encode_as_ids)
            self.write_mono_lines(recs, out_file)
            if args.get('text_files'):
                recs = self.read_mono_raw_data(raw_file, do_truncate, max_len, field.tokenize)
                self.write_mono_lines(recs, str(out_file).replace('.tsv', '.pieces.tsv'))

        _prep_file(args['mono_train_src'], self.mono_train_src, args['truncate'], args['src_len'],
                   self.src_vocab)
        _prep_file(args['mono_train_tgt'], self.mono_train_tgt, args['truncate'], args['tgt_len'],
                   self.tgt_vocab)

        _prep_file(args['mono_valid_src'], self.mono_valid_src, args['truncate'], args['src_len'],
                   self.src_vocab)
        _prep_file(args['mono_valid_tgt'], self.mono_valid_tgt, args['truncate'], args['tgt_len'],
                   self.tgt_vocab)

    def pre_process_finetune(self, args=None):
        """
        Pre process records for fine tuning
        :param args:
        :return:
        """
        log.info("Going to prep fine tune files")
        args = args if args else self.config['prep']
        assert 'finetune_src' in args
        assert 'finetune_tgt' in args
        # create Piece IDs
        finetune_recs = self.read_raw_data(args['finetune_src'], args['finetune_tgt'],
                                           args['truncate'], args['src_len'], args['tgt_len'],
                                           tokenizer=self.src_vocab.encode_as_ids)
        self.write_tsv(finetune_recs, self.finetune_file)

        if args.get('text_files'):
            # Redo again as plain text files
            finetune_recs = self.read_raw_data(args['finetune_src'], args['finetune_tgt'],
                                               args['truncate'], args['src_len'], args['tgt_len'],
                                               tokenizer=self.src_vocab.tokenize)
            self.write_tsv(finetune_recs, str(self.finetune_file).replace('.tsv', '.pieces.tsv'))

    def pre_process(self, args=None):
        args = args if args else self.config['prep']
        if self._unsupervised:
            self.pre_process_mono(args)
        else:
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
        args['src_vocab'] = len(self.src_vocab) if self.src_vocab else 0
        args['tgt_vocab'] = len(self.tgt_vocab) if self.tgt_vocab else 0
        self.config['updated_at'] = datetime.now().isoformat()
        self.store_config()

    def store_model(self, epoch: int, model, train_score: float, val_score: float, keep: int):
        """
        saves model to a given path
        :param epoch: epoch number of model
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
        name = f'model_{epoch:03d}_{train_score:.6f}_{val_score:.6f}.pkl'
        path = self.model_dir / name
        log.info(f"Saving epoch {epoch} to {path}")
        torch.save(model, str(path))

        for bad_model in self.list_models(sort='total_score', desc=False)[keep:]:
            log.info(f"Deleting bad model {bad_model} . Keep={keep}")
            os.remove(str(bad_model))

        with IO.writer(os.path.join(self.model_dir, 'scores.tsv'), append=True) as f:
            cols = [str(epoch), datetime.now().isoformat(), name, f'{train_score:g}',
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

    @property
    def src_vocab(self) -> Field:
        return self.shared_field if self.shared_field is not None else self.src_field

    @property
    def tgt_vocab(self):
        return self.shared_field if self.shared_field is not None else self.tgt_field

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
