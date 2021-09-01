import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, n=5):
    return np.convolve(x, np.ones(n)/n, 'valid')

data = {
    'Baseline':{
        'wr': np.load('running-third/8m-baseline_win_rates.npy'),
        'er': np.load('running-third/8m-baseline_episode_rewards.npy'),
        'kwargs': dict(linestyle='--', linewidth=2.5),
    },
    'Forward': {
        'wr': np.load('running-third/8m-forward_win_rates.npy'),
        'er': np.load('running-third/8m-forward_episode_rewards.npy'),
        'kwargs': dict(linestyle=':'),
    },
    'Forward VSN':{ # actually had 200k switch after parameter
        'wr': np.load('running-third/8m-forward-vsn_win_rates.npy'),
        'er': np.load('running-third/8m-forward-vsn_episode_rewards.npy'),
        'kwargs': dict(linestyle=':'),
    },
    'Sampling': {
        'wr': np.load('running-third/8m-sampling_win_rates.npy'),
        'er': np.load('running-third/8m-sampling_episode_rewards.npy'),
        'kwargs': dict(linestyle='-.'),
    },
    'Sampling VSN': { # actually had 200k switch after parameter
        'wr': np.load('running-third/8m-sampling-vsn_win_rates.npy'),
        'er': np.load('running-third/8m-sampling-vsn_episode_rewards.npy'),
        'kwargs': dict(linestyle='-.'),
    },
    'Sampling Switch': {
        'wr': np.load('running-third/8m-sampling-switch_win_rates.npy'),
        'er': np.load('running-third/8m-sampling-switch_episode_rewards.npy'),
        'kwargs': dict(linestyle='-.'),
    },
}

fig, (wr_ax, er_ax) = plt.subplots(2, 1)
fig.set_size_inches(10, 10)

for label in data:
    print(label)
    wr_ax.set_title('8m')
    wr_ax.plot(moving_average(data[label]['wr'][-401:]), label = label, **data[label].get('kwargs', {}))
    wr_ax.set_ylabel('Win rate')

    er_ax.plot(moving_average(data[label]['er'][-401:]), label = label, **data[label].get('kwargs', {}))
    er_ax.set_ylabel('Episode Reward')

wr_ax.legend()
er_ax.legend()
er_ax.set_xlabel('Evaluation iteration')
plt.tight_layout()
plt.show()

