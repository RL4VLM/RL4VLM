# Fine-tuning VLM with RL in the GymCards Environment

## Environment Setup

We suggest installing the python 3.10.0 environment:

```bash
conda create -n vrenv python=3.10.0
```

Then activate the environment and install the required packages:

```bash
conda activate vrenv
cd <path-to-this-repo>
pip install -e ../LLaVA
pip install -e ../gym-cards
pip install gymnasium[atari,accept-rom-license]
pip install stable-baselines3 wandb deepspeed sentencepiece git+https://github.com/openai/CLIP.git
pip install xformers
```

Note to follow the order of the installation commands above. Specifically, `LLaVA` should be installed first, and `xformers` should be installed at last.

## Reproduction

We currently support single-machine multi-GPU training. Specify the number of GPUs in `scripts/config_zero2`.

Numberline reproduction:

```bash
cd <path-to-this-repo>/VLM_PPO/scripts
bash run_numberline.sh
```

EZPoints reproduction:

```bash
cd <path-to-this-repo>/VLM_PPO/scripts
bash run_ezpoints.sh
```

ALFWorld reproduction: refer to the README file in the `alfworld` branch.

## Code structure

```
├── a2c_ppo_acktr
│   ├── algo
│   │   ├── __init__.py
│   │   └── ppo.py # main algorithm for training VLMs
│   ├── arguments.py # where we store all the arguments
│   ├── distributions.py
│   ├── envs.py # wrapper for the gymcards environment
│   ├── __init__.py
│   ├── llava_interface
│   │   ├── __init__.py
│   │   ├── interface.py # for generating text action and computing action log prob
│   │   └── utils.py # llava related utils
│   ├── model.py # for using VLM / MLLM as the backbone model
│   ├── rl_utils.py # for post-processing text action in each environment
│   ├── storage.py # for dealing with the replay buffer
│   └── utils.py
├── LICENSE
├── main.py # main run file
├── patch.py # for deepspeed accelerator
├── README.md
├── requirements.txt
└── scripts
    ├── config_zero2.yaml # the zero2 deepspeed config file
    ├── run_bj.sh
    ├── run_ezp.sh
    ├── run_nl.sh
    └── run_p24.sh

```

## Run Script Explanation
We provide some explanations for each argument in the from the [run_nl.sh](./scripts/run_nl.sh) as an example.

```
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0,1" accelerate launch --config_file config_zero2.yaml --main_process_port 29488 ../main.py \ # specifying the main process port for deepspeed, and the gpus for running the main file
    --env-name gym_cards/NumberLine-v0 \ # setting the main environment
    --init-lr 1e-5 \ # for cosine lr scheduler
    --end-lr 1e-9 \ # for cosine lr scheduler
    --lr_max_steps 25 \ # for cosine lr scheduler
    --eval-num-per-episode 200 \ # number of episodes for each evaluation
    --num-env-steps 15000 \ # total number of environment steps
    --num-steps 512 \ # number of environment steps for each on policy update
    --grad-accum-steps 128 \ # number of gradient accumulation per gradient update
    --max-new-tokens 256 \ # maximum number of tokens for open-ended text action
    --thought-prob-coef 0.5 \ # the scaling factor for the CoT tokens
    --use-gae \ # GAE estimator for the value function
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path /your_sft_checkpoint_for_numberline \ # your model path
    --use-lora \ # we only train our model using lora
    --train-vision all \ # decide which components from llava we want to train
    # --wandb-project you_wandb_proj \ wandb related
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4 # turn on for q4 quantization

```

## Acknowledgement
The backbone RL implementation is built upon [this repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).
