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
    for i in range(8):
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

        envs = [
            StarCraft2Env(map_name='3s_vs_3z', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir),
            StarCraft2Env(map_name='3s_vs_4z', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir),
            StarCraft2Env(map_name='3s_vs_5z', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir),
            # StarCraft2Env(map_name=args.map, step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir),
            # StarCraft2Env(map_name=args.map, step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir),
        ]

        # assume largest
        target_env = envs[-1]

        # change args to accommodate largest possible env
        for env in envs[-1:]:
            env_info = env.get_env_info()
            args.n_actions = env_info["n_actions"]
            args.n_agents = env_info["n_agents"]
            args.state_shape = env_info["state_shape"]
            args.obs_shape = env_info["obs_shape"]
            args.episode_limit = env_info["episode_limit"]

        # get feature section sizes;
        # move feats, enemy feats, ally feats, own feats, maybe time feat
        sections = [e.get_obs_sections() for e in envs]
        state_sections = envs[-1].get_state_sections()

        # create translators for state and features
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
