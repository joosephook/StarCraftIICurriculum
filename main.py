from common.replay_buffer import ReplayBuffer
from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args

import time
import os
import numpy as np

class Translator:
    def __init__(self, src, dst):
        assert len(src) == len(dst), f"Source and destination have different number of sections, {len(src)} vs {len(dst)}"
        self.src = src
        self.dst = dst

    def translate(self, vec):
        dst = np.zeros(self.dst[-1])

        for i in range(len(self.dst) - 1):
            width = self.src[i + 1] - self.src[i]
            dst[self.dst[i]:self.dst[i] + width] = vec[self.src[i]: self.src[i + 1]]

        return dst


class TranslatorMixin:
    def __init__(self, o_src, o_dst, s_src, s_dst):
        assert len(o_src) == len(o_dst), f"Source and destination observations have different number of sections, {len(o_src)} vs {len(o_dst)}"
        assert len(s_src) == len(s_dst), f"Source and destination states have different number of sections, {len(s_src)} vs {len(s_dst)}"
        self.obs_src = o_src
        self.obs_dst = o_dst
        self.state_src = s_src
        self.state_dst = s_dst

    def translate_obs(self, obs):
        return self._translate(obs, self.obs_src, self.obs_dst)

    def _translate(self, src, src_structure, target_structure):
        dst = np.zeros(target_structure[-1])

        for i in range(len(target_structure) - 1):
            width = src_structure[i + 1] - src_structure[i]
            dst[target_structure[i]:target_structure[i] + width] = src[src_structure[i]: src_structure[i + 1]]

        return dst

    def translate_state(self, state):
        return self._translate(state, self.state_src, self.state_dst)

def save_config(args):
    with open(__file__, 'r') as f:
        main = f.readlines()
    with open('common/arguments.py', 'r') as f:
        arguments = f.readlines()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    with open(args.save_path + f'/{os.path.basename(__file__)}', 'w') as f:
        f.writelines(main)
    with open(args.save_path + '/arguments.py', 'w') as f:
        f.writelines(arguments)

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

    for i in range(5):
        import torch
        seed = 12345+i
        np.random.seed(seed)
        torch.manual_seed(seed)

        map_names = ['3s_vs_5z']
        # envs = [
            # StarCraft2Env(map_name='5m_vs_6m', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
            #               seed=32**2-1),
            # StarCraft2Env(map_name='8m_vs_9m', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
            #               seed=32**2-1),
            # StarCraft2Env(map_name='10m_vs_11m', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
            #               seed=32**2-1),
            # StarCraft2Env(map_name='27m_vs_30m', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
            #               seed=32 ** 2 - 1),

            # StarCraft2Env(map_name='3s_vs_5z', step_mul=args.step_mul, difficulty=args.difficulty,
            #               game_version=args.game_version, replay_dir=args.replay_dir,
            #               seed=seed),
        # ]
        envs = [
            StarCraft2Env(map_name=m, step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
                          seed=seed)
            for m in map_names
        ]
        env_timesteps = [
            2_000_000
        ]

        curriculum = '->'.join(map_names)
        buffer_dtype = np.float16
        experiment_name = f'{int(time.time())}  {curriculum} {buffer_dtype} {args.alg}'

        args.save_path = os.path.join(args.result_dir, experiment_name, args.alg)
        save_config(args)

        with open(__file__, 'r') as f:
            main = f.readlines()
        with open('common/arguments.py', 'r') as f:
            arguments = f.readlines()

        # assume largest
        target_env = envs[-1]

        for env in envs:
            env_info = env.get_env_info()
            print(env_info)

        # change args to accommodate largest possible env
        for env in envs[-1:]:
            env_info = env.get_env_info()
            args.n_actions = env_info["n_actions"]
            args.n_agents = env_info["n_agents"]
            args.state_shape = env_info["state_shape"]
            args.obs_shape = env_info["obs_shape"]
            # TODO: what to do with episode limit???
            args.episode_limit = env_info["episode_limit"]

        # get feature section sizes;
        # move feats, enemy feats, ally feats, own feats, maybe time feat
        target_obs_sections   = envs[-1].get_obs_sections()
        target_state_sections = envs[-1].get_state_sections()

        obs_translators = []
        state_translators = []
        for env in envs:
            obs_translators.append(Translator(env.get_obs_sections(), target_obs_sections))
            state_translators.append(Translator(env.get_state_sections(), target_state_sections))

        # TODO: AGENT NUMBER IS BROKEN
        runner = Runner(envs[0], args, obs_translators[0], state_translators[0])
        new_buffer = True
        if not args.evaluate:
            runner.args.n_steps = 2_000_000
            runner.args.episode_limit = envs[0].get_env_info()["episode_limit"]
            if new_buffer and hasattr(runner, "buffer"):
                runner.buffer = ReplayBuffer(args, buffer_dtype)
            runner.run(i)
            runner.env.close()
            exit(0)

            # WHAT ABOUT EPSILON?
            runner.args.n_steps = 10_000
            runner.args.episode_limit = envs[1].get_env_info()["episode_limit"]
            runner.env = envs[1]
            runner.rolloutWorker.env = envs[1]
            runner.obs_trans = obs_translators[1]
            runner.state_trans = state_translators[1]
            if new_buffer and hasattr(runner, "buffer"):
                runner.buffer = ReplayBuffer(args, buffer_dtype)
            runner.run(i)

            # WHAT ABOUT EPSILON?
            runner.args.n_steps = 2_000_000
            runner.args.episode_limit = envs[2].get_env_info()["episode_limit"]
            runner.env = envs[2]
            runner.rolloutWorker.env = envs[2]
            runner.obs_trans = obs_translators[2]
            runner.state_trans = state_translators[2]
            if new_buffer and hasattr(runner, "buffer"):
                runner.buffer = ReplayBuffer(args, buffer_dtype)
            runner.run(i)

            # runner.args.n_steps = 2_000_000
            # runner.args.episode_limit = envs[3].get_env_info()["episode_limit"]
            # runner.env = envs[3]
            # runner.rolloutWorker.env = envs[3]
            # runner.obs_trans = obs_translators[3]
            # runner.state_trans = state_translators[3]
            # if new_buffer and hasattr(runner, "buffer"):
            #     runner.buffer = ReplayBuffer(args)
            # runner.run(i)

        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
