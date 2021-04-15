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


class EnvTransWrapper:
    def __init__(self, env, obs_trans, state_trans):
        self.env = env
        self.obs_trans = obs_trans
        self.state_trans = state_trans

    def __getattr__(self, item):
        if item == 'get_obs':
            return self.wrap_get_obs
        elif item == 'get_state':
            return self.wrap_get_state
        else:
            return getattr(self.env, item)

    def wrap_get_obs(self):
        observations = [self.obs_trans.translate(o) for o in self.env.get_obs()]
        return observations

    def wrap_get_state(self):
        state = self.state_trans.translate(self.env.get_state())
        return state


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

    timestamp = int(time.time())

    for i in range(5):
        import torch
        seed = 12345+i
        np.random.seed(seed)
        torch.manual_seed(seed)

        map_names = [
            '3s_vs_3z',
        ]
        envs = [
            StarCraft2Env(map_name=m, step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version, replay_dir=args.replay_dir,
                          seed=seed)
            for m in map_names
        ]
        env_timesteps = [
            200_000,
        ]

        for env in envs:
            env_info = env.get_env_info()
            print(env_info)

        curriculum = '->'.join(map_names)
        buffer_dtype = np.float16
        experiment_name = f'{timestamp} {curriculum} {buffer_dtype} {args.alg}'

        # assume largest
        target_env = envs[-1]
        target_env = StarCraft2Env(map_name='3s_vs_5z', step_mul=args.step_mul, difficulty=args.difficulty, game_version=args.game_version,
                                   replay_dir=args.replay_dir,
                                   seed=seed)
        # change args to accommodate largest possible env
        # assures the widths of the created neural networks are sufficient
        env_info = target_env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        # TODO: what to do with episode limit???
        args.episode_limit = env_info["episode_limit"]

        def create_translators(envs, target_env):
            target_obs_sections   = target_env.get_obs_sections()
            target_state_sections = target_env.get_state_sections()

            wrapped_envs = []
            for env in envs:
                o_trans = Translator(env.get_obs_sections(), target_obs_sections)
                s_trans = Translator(env.get_state_sections(), target_state_sections)
                wrapped_envs.append(EnvTransWrapper(env, o_trans, s_trans))

            return wrapped_envs

        envs = create_translators(envs, target_env)
        # TODO: AGENT NUMBER IS BROKEN
        args.save_path = os.path.join(args.result_dir, experiment_name, args.alg, str(i))
        save_config(args)
        runner = Runner(envs[0], args, target_env)

        new_buffer = True
        assert len(envs) == len(env_timesteps)
        episode_limits = [ e.get_env_info()["episode_limit"] for e in envs]

        if not args.evaluate:
            for env, env_time, ep_lim, in zip(
                envs, env_timesteps, episode_limits
            ):
                runner.env = env
                runner.rolloutWorker.env = env
                runner.args.n_steps = env_time
                runner.args.episode_limit = ep_lim
                if new_buffer and hasattr(runner, "buffer"):
                    runner.buffer = ReplayBuffer(args, buffer_dtype)
                runner.run(i)
                runner.env.close()

        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
