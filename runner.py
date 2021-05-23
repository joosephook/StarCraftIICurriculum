from collections import defaultdict

import numpy as np
import logging
import os

from torch.utils.tensorboard import SummaryWriter

from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import io


class Runner:
    def __init__(self, curriculum, args, target_env):
        self.target_env = target_env
        self.curriculum = curriculum

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(None, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(None, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find(
                'reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = None
        self.args = args
        self.win_rates = []
        self.eval_episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.train_rewards = []
        self.ratios = []
        self.historical_params = {}
        self.switch = True  # we will be switching to some task
        self.patience = 20
        self.writer: SummaryWriter = None
        self.eval_envs = None

    def run(self):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while True:
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, eval_episode_reward = self.evaluate(time_steps, self.target_env)
                self.win_rates.append(win_rate)
                self.eval_episode_rewards.append(eval_episode_reward)
                self.plt()
                evaluate_steps += 1

                performance = int(eval_episode_reward)
                self.curriculum.update(performance, self.agents, time_steps, train_steps)

                # eval in other envs
                for env in self.eval_envs:
                    self.evaluate(time_steps, env)

            try:
                env = self.curriculum.get()
                buffer = env.buffer
                self.rolloutWorker.env = env
                logging.info("Restoring map {}".format(self.rolloutWorker.env.map_name))
            except IndexError:  # done
                self.agents.policy.save_model(train_step)
                self.plt()
                break

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, train_episode_reward, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                self.train_rewards.append(train_episode_reward)
                episodes.append(episode)
                time_steps += steps

            logging.info('Time_steps {}, train_episode_reward {}'.format(time_steps, train_episode_reward))
            # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find(
                    'reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = buffer.sample(min(buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

            self.writer.add_scalar(f'Reward/train/', train_episode_reward, global_step=time_steps)
            self.writer.add_scalar(f'Reward/train/{env.map_name}', train_episode_reward, global_step=time_steps)
            for n, p in self.agents.policy.eval_rnn.named_parameters():
                self.writer.add_scalar(f'eval_rnn/{n}/norm', p.norm(), global_step=time_steps)
                self.writer.add_scalar(f'eval_rnn/grad/{n}/norm', p.grad.norm(), global_step=time_steps)
                self.writer.add_scalar(f'eval_rnn/{n}/norm/{env.map_name}', p.norm(), global_step=time_steps)
                self.writer.add_scalar(f'eval_rnn/grad/{n}/norm/{env.map_name}', p.grad.norm(), global_step=time_steps)
            for n, p in self.agents.policy.eval_qmix_net.named_parameters():
                self.writer.add_scalar(f'eval_qmix_net/{n}/norm', p.norm(), global_step=time_steps)
                self.writer.add_scalar(f'eval_qmix_net/grad/{n}/norm', p.grad.norm(), global_step=time_steps)
                self.writer.add_scalar(f'eval_qmix_net/{n}/norm/{env.map_name}', p.norm(), global_step=time_steps)
                self.writer.add_scalar(f'eval_qmix_net/grad/{n}/norm/{env.map_name}', p.grad.norm(), global_step=time_steps)

    def evaluate(self, time_steps, env):
        win_number = 0
        episode_rewards = 0
        self.rolloutWorker.env = env
        logging.info("Evaluating in map {}".format(self.rolloutWorker.env.map_name))
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            logging.info('Eval_epoch {}, eval_episode_reward {}'.format(epoch, episode_reward))
            episode_rewards += episode_reward
            self.writer.add_scalar(f'Reward/eval/{self.rolloutWorker.env.map_name}', episode_reward, time_steps + epoch)
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self):
        plt.figure().set_size_inches(10, 15)
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(3, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(3, 1, 2)
        plt.plot(range(len(self.eval_episode_rewards)), self.eval_episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('eval_episode_rewards')

        plt.subplot(3, 1, 3)
        train_rewards = np.array_split(self.train_rewards, len(self.eval_episode_rewards))
        mean_train_rewards = [np.mean(t) for t in train_rewards]
        plt.plot(range(len((mean_train_rewards))), mean_train_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('train_episode_rewards')

        plt.tight_layout()
        plt.savefig(self.save_path + '/plt.png', format='png')
        np.save(self.save_path + '/win_rates', self.win_rates)
        np.save(self.save_path + '/eval_rewards', self.eval_episode_rewards)
        np.save(self.save_path + '/train_rewards', self.train_rewards)
        plt.close()
