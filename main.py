import json

from torch.utils.tensorboard import SummaryWriter

from common.replay_buffer import ReplayBuffer
from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args

import time
import os
import numpy as np
import logging


def save_config(args):
    import shutil

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    with open(args.save_path + f'/{os.path.basename(__file__)}', 'w') as f, open(__file__, 'r') as fin:
        f.writelines(fin.readlines())
    with open(args.save_path + '/arguments.py', 'w') as f, open('common/arguments.py', 'r') as fin:
        f.writelines(fin.readlines())
    with open(args.save_path + f'/{args.config}', 'w') as f, open(args.config, 'r') as fin:
        f.writelines(fin.readlines())

    for f in 'common network agent policy'.split():
        shutil.copytree(f, os.path.join(args.save_path, f))
    for f in 'main.py runner.py'.split():
        shutil.copy(f, os.path.join(args.save_path, f))


if __name__ == '__main__':
    args = get_common_args()
    args.alg = 'qmix'
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    unsupported = "coma central_v reinforce".split()
    if args.alg in unsupported:
        assert False, "These algos aren't supported yet (refer to ReplayBuffer)"  # on-policy algorithms, don't utilise the replaybuffer

    timestamp = int(time.time())
    i = args.i

    import torch
    seed = 12345+i
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(args.config, 'r') as f:
        config = json.load(f)

    assert all(["map_names" in config, "map_timesteps" in config, "target_map" in config]), "Check config file"
    assert len(config["map_names"]) == len(config["map_timesteps"]), "Check config file"

    conf_name, __ = args.config.split('.')
    experiment_name = f'{timestamp} {args.alg} {i}'
    args.save_path = os.path.join(conf_name, experiment_name)
    save_config(args)
    logging.basicConfig(filename=os.path.join(args.save_path, 'out.log'), level=logging.INFO)

    if config.get("difficulties"):
        difficulties =  config["difficulties"]
    else:
        difficulties = [args.difficulty]*len(config["map_names"])

    target_env = StarCraft2Env(map_name=config["target_map"],
                               step_mul=args.step_mul,
                               difficulty=args.difficulty,
                               game_version=args.game_version,
                               replay_dir=args.replay_dir,
                               seed=seed,
                               shuffle=False,
                               )
    eval_envs = [
        StarCraft2Env(map_name=m,
                      step_mul=args.step_mul,
                      difficulty=d,
                      game_version=args.game_version,
                      replay_dir=args.replay_dir,
                      seed=seed,
                      pad_agents=target_env.n_agents,
                      pad_enemies=target_env.n_enemies,
                      shuffle=False,
                      )
        for m, d in zip(config["eval_maps"], difficulties)
    ]
    train_envs = [
        StarCraft2Env(map_name=m,
                      step_mul=args.step_mul,
                      difficulty=d,
                      game_version=args.game_version,
                      replay_dir=args.replay_dir,
                      seed=seed,
                      pad_agents=target_env.n_agents,
                      pad_enemies=target_env.n_enemies,
                      shuffle=False,
                      )

        for m, d in zip(config["map_names"], difficulties)
    ]

    for env in train_envs:
        env_info = env.get_env_info()
        logging.info(env_info)

    # change args to accommodate largest possible env
    # assures the widths of the created neural networks are sufficient
    env_info = target_env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    runner = Runner(train_envs[0], args, target_env)
    runner.eval_envs = eval_envs

    new_buffer = True
    if not args.evaluate:
        for env, env_time in zip(train_envs, config["map_timesteps"]):
            runner.train_env = env
            runner.args.n_steps = env_time
            runner.writer = SummaryWriter(os.path.join(args.save_path, env.map_name))
            args.episode_limit = env.get_env_info()["episode_limit"]
            runner.switch = env.map_name != train_envs[-1].map_name

            if new_buffer and hasattr(runner, "buffer"):
                env_info = env.get_env_info()
                target_info = target_env.get_env_info()
                runner.buffer = ReplayBuffer(
                    n_actions=target_info['n_actions'],
                    n_agents=env_info['n_agents'],
                    obs_shape=target_info['obs_shape'],
                    state_shape=target_info['state_shape'],
                    episode_limit=env_info['episode_limit'],
                    size=args.buffer_size,
                    alg=args.alg,
                    noise_dim=args.noise_dim,
                    dtype=np.float16,
                )

            runner.patience = 20
            runner.run(i)
            runner.rolloutWorker.epsilon = args.epsilon
            runner.agents.policy.reset_optimiser()
            runner.agents.policy.load_target()
            runner.train_env.close()

    else:
        win_rate, _ = runner.evaluate()
        print('The win rate of {} is  {}'.format(args.alg, win_rate))
    env.close()
