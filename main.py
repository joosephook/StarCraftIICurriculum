import io
import json
from typing import Dict, List, Tuple

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


class ForwardCurriculum:
    def __init__(self, envs: List[Tuple[StarCraft2Env, int]], patience, switch_callback):
        self.envs = envs
        self._patience = patience
        self.patience = None
        self.current_env = None
        self.current_max_steps = None
        self.historical_params = None
        self.no_improvement = None
        self.switch_callback = switch_callback
        for e, _ in envs:
            e.switch = e.map_name != envs[-1][0].map_name

        self.next()

    def get(self):
        if self.current_env.total_steps >= self.current_max_steps:
            self.next()
        return self.current_env

    def next(self):
        if self.current_env:
            self.current_env.close()
        self.current_env, self.current_max_steps = self.envs.pop(0)
        self.historical_params = {}
        self.no_improvement = 0
        self.patience = self._patience
        self.switch_callback()

    def update(self, performance, agents, time_steps, train_steps):
        if self.current_env.switch:
            if len(self.historical_params) == 0:
                logging.info("First params eval perf @ {} timesteps: {}".format(time_steps, performance))
                # save weights when empty
                buf = io.BytesIO()
                agents.policy.save(buf)
                buf.seek(0)
                self.historical_params[performance] = buf
            elif performance > max(self.historical_params):
                logging.info("New best eval perf @ {} timesteps: {}".format(time_steps, performance))
                # save weights when get better performance
                buf = io.BytesIO()
                agents.policy.save(buf)
                buf.seek(0)
                self.historical_params[performance] = buf
                self.no_improvement = 0
            elif self.no_improvement < self.patience:
                self.no_improvement += 1
                logging.info("No improvement: {}".format(self.no_improvement))
            elif self.no_improvement >= self.patience:
                logging.info("Switching to next task @ {} timesteps".format(time_steps))
                best_key = max(self.historical_params)
                buf = self.historical_params[best_key]
                agents.policy.load(buf)
                buf.seek(0)
                agents.policy.save_model(train_steps)
                self.next()


class SamplingCurriculum:
    def __init__(self, envs: List[StarCraft2Env], p=[], total_timesteps=None):
        self.envs = envs
        assert len(p) == len(envs), "Need prob for each env"
        assert np.sum(p) == 1.0, "Probs must sum to 1"
        self.p = np.array(p)
        self.rng = np.random.default_rng(0)
        self.total_timesteps = total_timesteps
        self.current_timesteps = 0

    def get(self):
        self.current_timesteps = sum(e.total_steps for e in self.envs)
        if self.current_timesteps >= self.total_timesteps:
            raise IndexError(f"Trained for {self.current_timesteps}")

        idx = self.rng.choice(np.arange(len(self.envs)), 1, p=self.p)[0]
        return self.envs[idx]

    def update(self, *args, **kwargs):
        pass

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

    assert all(["map_names" in config, "target_map" in config]), "Check config file"
    if config.get("map_timesteps", None):
        assert len(config["map_timesteps"]) == len(config["map_names"]), "Check map timesteps and map names"

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
                      noise=config.get("noise", None) if m != config["target_map"] else None
                      )

        for m, d in zip(config["map_names"], difficulties)
    ]

    for env in train_envs:
        env_info = env.get_env_info()
        target_info = target_env.get_env_info()
        env.buffer = ReplayBuffer(
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
        logging.info(env_info)
    # change args to accommodate largest possible env
    # assures the widths of the created neural networks are sufficient
    env_info = target_env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    runner = Runner(None, args, target_env)

    def switch_callback():
        runner.rolloutWorker.epsilon = args.epsilon
        runner.agents.policy.reset_optimiser()
        runner.agents.policy.load_target()

    if config.get("probs", None) and config.get("total_timesteps", None):
        curriculum = SamplingCurriculum(
            [
                env for env in train_envs
            ],
            total_timesteps=config["total_timesteps"],
            p=[0.2, 0.8]
        )
    else:
        curriculum = ForwardCurriculum(
            [
                (env, steps) for env, steps in zip(train_envs, config["map_timesteps"])
            ],
            patience=config.get("patience", 20),
            switch_callback=switch_callback
        )



    runner.curriculum = curriculum
    runner.eval_envs = eval_envs
    runner.writer = SummaryWriter(args.save_path)


    if not args.evaluate:
        runner.run()
    else:
        win_rate, _ = runner.evaluate()
        print('The win rate of {} is  {}'.format(args.alg, win_rate))
    env.close()
