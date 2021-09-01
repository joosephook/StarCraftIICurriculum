from smac.env.multiagentenv import MultiAgentEnv
from ma_gym.envs.traffic_junction_hard.traffic_junction_hard import TrafficJunctionHard
from typing import Dict, List, Tuple, NewType
import numpy as np

# Type aliases
Obs =  NewType('Obs', np.ndarray)
State =      NewType('State', np.ndarray)
Action =     NewType('Action', np.ndarray)
Terminated = NewType('Terminated', bool)
Reward =     NewType('Reward', float)
Info =       NewType('Info', dict)
OneEncoded = NewType('OneEncoded', np.ndarray)


class MAGymWrapper(MultiAgentEnv):
    def __init__(self, env: TrafficJunctionHard, fake_agents):
        self.env = env
        self.fake_agents = fake_agents
        self.env_args = dict(
            grid_shape=env._grid_shape,
            step_cost=env._step_cost,
            n_agents=env.n_agents,
            rcoll=env._collision_reward,
            arrive_prob=env._arrive_prob,
            full_observable=env.full_observable
        )
        self.map_name = f'TrafficJuncionHard({self.env_args})'
        self.env.reset() # populate necessary data structures
        self.last_action = np.zeros((self.fake_agents, self.get_total_actions()), dtype=np.int)
        self.n_agents = env.n_agents
        self.agents = range(self.n_agents)
        self.n_actions = self.get_total_actions()
        self.episode_limit = self.env._max_steps
        self.total_steps = 0
        self.obs_size = self.get_obs_size()
        self.state_size = self.get_state_size()

    def reset(self)-> Tuple[List[Obs], State]:
        self.total_steps += self.env._step_count
        observations = self.env.reset()
        return list(map(np.asarray, observations)), np.asarray(self.env.get_agent_obs())

    def step(self, actions)-> Tuple[Reward, Terminated, Info]:
        observations, rewards, dones, info = self.env.step(actions)
        self.last_action *= 0
        for i, a in enumerate(actions):
            self.last_action[i, a] = 1
        return Reward(sum(rewards)), Terminated(all(dones)), Info({})

    def get_obs(self) -> List[Obs]:
        return [np.asarray(self.get_obs_agent(i)) for i in range(self.env.n_agents)]

    def get_obs_agent(self, agent_id) -> Obs:
        padded = np.zeros((*self.env._agent_view_mask, self.fake_agents+2+self.env._n_routes))
        obs = np.asarray(self.env.get_agent_obs()[agent_id]).reshape((*self.env._agent_view_mask, self.env.n_agents+2+2))
        padded[:, :, :self.env.n_agents] = obs[:, :, :self.env.n_agents]
        padded[:, :, -(2+2):] = obs[:, :, self.env.n_agents:]
        return padded.flatten()

    def get_obs_size(self) -> int:
        return self.get_obs()[0].shape[-1]

    def get_state(self) -> State:
        padded = np.zeros(self.fake_agents*self.obs_size)
        state = np.asarray(self.get_obs()).ravel()
        padded[:len(state)] = state
        return padded

    def get_state_size(self) -> int:
        return np.asarray(self.get_state()).ravel().shape[-1]

    def get_avail_actions(self) -> List[OneEncoded]:
        return [self.get_avail_agent_actions(i) for i in range(self.env.n_agents)]

    def get_avail_agent_actions(self, agent_id) -> OneEncoded:
        return self.env.get_avail_actions_agent(agent_id)

    def get_total_actions(self) -> int: # number of actions
        return self.env.action_space._agents_action_space[0].n

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        return dict(
            n_actions=self.env.action_space._agents_action_space[0].n,
            n_agents=self.env.n_agents,
            state_shape=self.get_state_size(),
            obs_shape=self.get_obs_size(),
            episode_limit=self.env._max_steps
        )

