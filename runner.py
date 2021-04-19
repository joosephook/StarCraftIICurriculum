import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import io


class Runner:
    def __init__(self, env, args, target_env):
        self.train_env = env
        self.target_env = target_env

        self.env = self.train_env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = None
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.train_rewards = []
        self.eval_rewards  = []
        self.ratios = []
        self.historical_params = {}
        self.switch = True # we will be switching to some task
        self.patience = 5

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1

        no_insert = 0

        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, eval_episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(eval_episode_reward)
                self.plt(num)
                evaluate_steps += 1

                key = int(eval_episode_reward)
                if self.switch and len(self.historical_params) == 0:
                    # save weights when empty
                    buf = io.BytesIO()
                    self.agents.policy.save(buf)
                    buf.seek(0)
                    self.historical_params[key] = buf
                elif self.switch and key > max(self.historical_params):
                    # save weights when get better performance
                    buf = io.BytesIO()
                    self.agents.policy.save(buf)
                    buf.seek(0)
                    self.historical_params[key] = buf
                    no_insert = 0
                elif self.switch:
                    no_insert += 1

                if self.switch and no_insert > self.patience:
                    best_key = max(self.historical_params)
                    self.agents.policy.load(self.historical_params[best_key])
                    break

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, train_episode_reward, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                self.train_rewards.append(train_episode_reward)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.agents.policy.save_model(train_step)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        self.rolloutWorker.env = self.target_env
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            self.eval_rewards.append(episode_reward)
            if win_tag:
                win_number += 1
        self.rolloutWorker.env = self.train_env
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure().set_size_inches(10, 15)
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(4, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(4, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('eval_episode_rewards')


        plt.subplot(4, 1, 3)
        train_rewards = np.array_split(self.train_rewards,len(self.episode_rewards))
        mean_train_rewards = [np.mean(t) for t in train_rewards]
        plt.plot(range(len((mean_train_rewards))), mean_train_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('train_episode_rewards')

        past_train = self.train_rewards[-2*self.args.evaluate_epoch:-self.args.evaluate_epoch]
        past_eval = self.train_rewards[-2*self.args.evaluate_epoch:-self.args.evaluate_epoch]
        latest_train = self.train_rewards[-self.args.evaluate_epoch:]
        latest_eval = self.eval_rewards[-self.args.evaluate_epoch:]

        def iqr(data, epsilon=1e-5):
            # avoid division by zero
            return np.subtract(*np.percentile(data, [75, 25])) + epsilon

        if len(self.train_rewards) >= 2*self.args.evaluate_epoch and \
           len(self.eval_rewards) >= 2*self.args.evaluate_epoch:
            plt.subplot(4, 1, 4)
            train_iqr_ratio = iqr(latest_train)/iqr(past_train)
            eval_iqr_ratio = iqr(latest_eval)/iqr(past_eval)
            self.ratios.append(train_iqr_ratio/eval_iqr_ratio)
            plt.plot(range(len(self.ratios)), self.ratios)
            plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
            plt.ylabel('train_IQR')

        plt.tight_layout()
        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()









