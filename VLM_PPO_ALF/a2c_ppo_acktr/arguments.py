import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--init-lr',
        type=float,
        default=1e-6,
        help='initial learning rate (default: 1e-6)')
    parser.add_argument(
        '--end-lr',
        type=float,
        default=1e-8,
        help='final learning rate (default: 1e-8)')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0,
        help='weight decay (default: 1e-3)')
    parser.add_argument(
        '--explore_portion',
        type=float,
        default=0.1,
        help='rate of exporation, updates, a number between 0-1')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-7,
        help='RMSprop optimizer epsilon (default: 1e-7)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9,
        help='discount factor for rewards (default: 0.9)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.01,
        help='max norm of gradients (default: 0.01)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=256,
        help='number of environment steps collected at each iteration (default: 256)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--grad-accum-steps',
        type=int,
        default=2,
        help='the number of gradient accumulation steps (default: 2)')
    parser.add_argument(
        '--mini-batch-size',
        type=int,
        default=1,
        help='size of mini-batches for each update (default: 1)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.1,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='gym_cards/Blackjack-v0',
        help='environment to train on (default: gym_cards/Blackjack-v0)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--eval-num-per-episode',
        type=int,
        default=100,
        help='number of episodes to evaluate the agent (default: 100)')
    parser.add_argument(
        '--lr_max_steps',
        type=int,
        default=100,
        help='number of steps for lr scheduler (default: 100)')
    # arguments for llava interface
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--pretrain_mm_adapter", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="What is the next action? Please format your response as 'The next action is \{response\}.'")
    parser.add_argument("--data-path", type=str, default="../gym-cards/bc_data.json")
    parser.add_argument("--image-folder", type=str, default="../gym-cards/images")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use-lora", default=False, action='store_true')
    parser.add_argument("--train-vision", type=str, default='all')
    parser.add_argument("--thought-prob-coef", type=float, default=1.0, help='any number between 0-1, multiplier for the log thought probability')
    parser.add_argument("--action_only_prompt", default=False, action='store_true')
    # Argments for supporting alf config file
    parser.add_argument("--alf_config", type=str, default=None)

    # arguments for wandb
    parser.add_argument("--use-wandb", default=False, action='store_true')
    parser.add_argument("--wandb-project", type=str, default='test')
    parser.add_argument("--wandb-run", type=str, default='test')
    parser.add_argument("--q4", default=False, action='store_true')
    parser.add_argument("--q8", default=False, action='store_true')
    args = parser.parse_args()


    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
