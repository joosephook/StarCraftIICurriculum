import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, n_actions=None, n_agents=None, state_shape=None, obs_shape=None, size=None, episode_limit=None, dtype=None, alg=None, noise_dim=None):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.size = size
        self.episode_limit = episode_limit
        self.alg = alg

        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        assert True, "Really think about the buffer thing: does zero padding"\
                      "make sense for ALL of these? It makes sense for inputs"\
                      "to the network, but does it make sense for the action?"\
                      "The other agents aren't always taking the 0 action, so it"\
                      " would be wrong to say they do..."
        self.buffers = {
            # zero padding ok because it is just input data, 0 means "missing data"
            'o': np.zeros([self.size, self.episode_limit, self.n_agents, self.obs_shape], dtype=dtype),
            # action: zero padding doesn't make sense because the non-existent agents would be taking action 0,
            'u': np.zeros([self.size, self.episode_limit, self.n_agents, 1], dtype=dtype),
            # zero padding ok because it is just input data, 0 means "missing data"
            's': np.zeros([self.size, self.episode_limit, self.state_shape], dtype=dtype),
            # no zero padding needed
            'r': np.zeros([self.size, self.episode_limit, 1], dtype=dtype),
            # zero padding ok because it is just input data, 0 means "missing data"
            'o_next': np.zeros([self.size, self.episode_limit, self.n_agents, self.obs_shape], dtype=dtype),
            # zero padding ok because it is just input data, 0 means "missing data"
            's_next': np.zeros([self.size, self.episode_limit, self.state_shape], dtype=dtype),
            # zero padding would be ok because avail_u is the mask of available actions, zero means an action is not available
            'avail_u': np.zeros([self.size, self.episode_limit, self.n_agents, self.n_actions], dtype=dtype),
            # zero padding would be ok because avail_u_next is the mask of available actions, zero means an action is not available
            'avail_u_next': np.zeros([self.size, self.episode_limit, self.n_agents, self.n_actions], dtype=dtype),
            # zero padding would be ok because adding more zeroes just extends the space of possible actions
            'u_onehot': np.zeros([self.size, self.episode_limit, self.n_agents, self.n_actions], dtype=dtype),
            # zero padding not needed
            'padded': np.zeros([self.size, self.episode_limit, 1], dtype=dtype),
            # zero padding not needed
            'terminated': np.zeros([self.size, self.episode_limit, 1], dtype=dtype)
        }
        # force allocation, we don't want to run OOM later in the experiment...
        bytes = 0
        for buf in self.buffers.values():
            # buf[:] = 0
            bytes += buf.nbytes
        print("Buffers would consume", bytes/(1024*1024*1024), "GiB of memory.")


        if self.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.noise_dim], dtype=dtype)
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size, time_limit, n_agents, _ = episode_batch['o'].shape  # episode_number
        batch_size, time_limit, _, n_actions = episode_batch['avail_u'].shape

        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs, :, :n_agents, :] = episode_batch['o']
            self.buffers['u'][idxs, :, :n_agents, :] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs, :, :n_agents, :] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs, :, :n_agents, :n_actions] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs, :, :n_agents, :n_actions] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs, :, :n_agents, :n_actions] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
