from common.replay_buffer import ReplayBuffer
from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args

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

if __name__ == '__main__':
    for i in range(1):
        args = get_common_args()
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

        import torch
        import numpy
        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)
        envs = [
            StarCraft2Env(map_name='3s_vs_3z', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
                          seed=32**2-1),
            StarCraft2Env(map_name='3s_vs_4z', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
                          seed=32**2-1),
            StarCraft2Env(map_name='3s_vs_5z', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
                          seed=32**2-1),
            # StarCraft2Env(map_name=args.map, step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir),
            # StarCraft2Env(map_name=args.map, step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir),
        ]

        import time
        import os
        args.map = f'TEST CLEARBUF{int(time.time())}'
        with open(__file__, 'r') as f:
            main = f.readlines()
        with open('common/arguments.py', 'r') as f:
            arguments = f.readlines()

        path = args.result_dir + '/' + args.alg + '/' + args.map

        if not os.path.isdir(path):
            os.makedirs(path)

        with open(path+f'/{os.path.basename(__file__)}', 'w') as f:
            f.writelines(main)
        with open(path+'/arguments.py', 'w') as f:
            f.writelines(arguments)

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

        # create translators for state and features
        runner = Runner(envs[0], args, obs_translators[0], state_translators[0])
        if not args.evaluate:
            runner.args.n_steps = 20_000
            runner.run(i)
            runner.env.close()

            # WHAT ABOUT EPSILON?
            runner.args.n_steps = 20_000
            runner.env = envs[1]
            runner.rolloutWorker.env = envs[1]
            runner.obs_trans = obs_translators[1]
            runner.state_trans = state_translators[1]
            if True and not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find(
                    'reinforce') == -1:  # these 3 algorithms are on-poliy
                runner.buffer = ReplayBuffer(args)
            runner.run(i)

            runner.args.n_steps = 2_000_000
            runner.env = envs[2]
            runner.rolloutWorker.env = envs[2]
            runner.obs_trans = obs_translators[2]
            runner.state_trans = state_translators[2]
            if True and not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find(
                    'reinforce') == -1:  # these 3 algorithms are on-poliy
                runner.buffer = ReplayBuffer(args)
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
