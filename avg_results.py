import sys
import numpy as np
import os
from pathlib import Path


for directory in sys.argv[1:]:
    indir = Path(directory).name
    
    win_rates = []
    episode_rewards = []
    
    for root, dir, files in os.walk(indir):
        for f in files:
            if '.npy' in f:
                if 'win_rates' in f:
                    data = np.load(os.path.join(root, f))
                    print(os.path.join(root, f), 'win_rates', data.shape)
                    win_rates.append(data[-401:])
                if 'episode_rewards' in f or 'eval_rewards' in f:
                    data = np.load(os.path.join(root, f))
                    episode_rewards.append(data[-401:])
                    print(os.path.join(root, f), 'episode_rewards', data.shape)
    
    try:
        if win_rates and episode_rewards:
            win_rates = np.array(win_rates).mean(axis=0)
            episode_rewards = np.array(episode_rewards).mean(axis=0)
            np.save(f'{indir}_win_rates.npy', win_rates)
            np.save(f'{indir}_episode_rewards.npy', episode_rewards)
    except Exception as e:
        print(indir, e)
