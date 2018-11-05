# CLI interface to decode task
import argparse
import sys
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
import torch

from virtchar import DialogExperiment as Experiment, log
from virtchar.use.decoder import Decoder, ReloadEvent


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.decode", description="Decode using NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument("model_path", type=str, nargs='*',
                        help="Path to model's checkpoint. "
                             "If not specified, a best model (based on the score on validation set)"
                             " from the experiment directory will be used."
                             " If multiple paths are specified, then an ensembling is performed by"
                             " averaging the param weights")
    parser.add_argument("-if", '--input', default=sys.stdin,
                        type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                        help='Input file path. default is STDIN')
    parser.add_argument("-of", '--output', default=sys.stdout,
                        type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        help='Output File path. default is STDOUT')
    parser.add_argument("-bs", '--beam-size', type=int, default=5,
                        help='Beam size. beam_size=1 is greedy, '
                             'In theory: higher beam is better approximation but expensive. '
                             'But in practice, higher beam doesnt always increase.')
    parser.add_argument("-ml", '--max-len', type=int, default=100,
                        help='Maximum output sequence length')
    parser.add_argument("-nh", '--num-hyp', type=int, default=1,
                        help='Number of hypothesis to output. This should be smaller than beam_size')
    parser.add_argument("--prepared", dest="prepared", action='store_true',
                        help='Each token is a valid integer which is an index to embedding,'
                             ' so skip indexifying again')
    parser.add_argument("-it", '--interactive', action='store_true',
                        help='Open interactive shell with decoder')
    parser.add_argument("-sc", '--skip-check', action='store_true',
                        help='Skip Checking whether the experiment dir is prepared and trained')

    parser.add_argument("-en", '--ensemble', type=int, default=1,
                        help='Ensemble best --ensemble models by averaging them')
    return vars(parser.parse_args())


def main():
    # No grads required
    torch.set_grad_enabled(False)
    args = parse_args()
    gen_args = {}

    exp = Experiment(args.pop('work_dir'), read_only=True)

    if not args.pop('skip_check'):  # if --skip-check is not requested
        assert exp.has_prepared(),\
            f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        assert exp.has_trained(),\
            f'Experiment dir {exp.work_dir} is not ready to decode. Please run "train" sub task'

    decoder = Decoder.new(exp, gen_args=gen_args, model_paths=args.pop('model_path', None),
                          ensemble=args.pop('ensemble', 1))
    if args.pop('interactive'):
        if args['input'] != sys.stdin or args['output'] != sys.stdout:
            log.warning('--input and --output args are not applicable in --interactive mode')
        args.pop('input')
        args.pop('output')

        while True:
            try:
                # an hacky way to unload and reload model when user tries to switch models
                decoder.decode_interactive(**args)
                break  # exit loop if there is no request for reload
            except ReloadEvent as re:
                decoder = Decoder.new(exp, gen_args=gen_args, model_paths=re.model_paths)
                args = re.state
                # go back to loop and redo interactive shell
    else:
        return decoder.decode_file(args.pop('input'), args.pop('output'), **args)


if __name__ == '__main__':
    main()
