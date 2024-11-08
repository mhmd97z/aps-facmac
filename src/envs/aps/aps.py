import gym
import torch
from gym import spaces
from envs.aps.lib.network_simlator import NetworkSimulator
from envs.aps.lib.data_store import DataStore
from envs.aps.lib.utils import get_polar, range_normalization, tpdv_parse


class Aps(gym.Env):
    def __init__(self, env_args=None, args=None):
        self.env_args = env_args
        tpdv_parse(self.env_args)
        self.simulator = NetworkSimulator(env_args['simulation_scenario'])
        self.history_length = self.env_args['history_length']
        self.datastore = DataStore(self.history_length, ['obs'])

        num_ues = self.simulator.scenario_conf['number_of_ues']
        num_aps = self.simulator.scenario_conf['number_of_aps']
        if self.env_args['use_gnn_embedding'] and \
            self.env_args['simulation_scenario']['precoding_algorithm'] == 'olp':
            self.feature_length = self.env_args['embedding_length'] + 1
        else:
            self.feature_length = 3

        self.n_agents = num_ues * num_aps

        self.action_space = [spaces.Discrete(2) for _ in range(self.n_agents)]
        self.observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]
        self.share_observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.n_agents * self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]

        self.episode_limit = self.env_args['episode_limit']


    def step(self, actions):
        # actions = (actions >= 0).int()
        self.simulator.step(actions)

        obs, state, reward, mask, info = self.compute_state_reward()
        done = False

        # return obs, state, reward, mask, done, info
        return reward, done, info


    def compute_state_reward(self):
        # state calc
        simulator_info = self.simulator.datastore.get_last_k_elements()
        serving_mask = self.simulator.serving_mask.clone().detach().flatten().to(torch.int32)

        if self.env_args['use_gnn_embedding'] and \
            self.env_args['simulation_scenario']['precoding_algorithm'] == 'olp':
            embedding = torch.mean(
                simulator_info['embedding'].clone().detach(), 
                axis=0)
            obs = torch.cat((embedding, serving_mask.unsqueeze(0)), dim=0)
        else:
            channel_coef = torch.mean(
                simulator_info['channel_coef'].clone().detach(), axis=0
                ).flatten()
            chan_magnitude, chan_phase = get_polar(channel_coef)
            obs = torch.cat(
                (chan_magnitude.unsqueeze(0), chan_phase.unsqueeze(0), 
                 serving_mask.unsqueeze(0)), 
                dim=0)

        # to get the history of state variables
        self.datastore.add(obs=obs)
        state = self.datastore.get_last_k_elements()['obs'].permute(2, 0, 1)
        obs = state.clone().detach()
        state = state.unsqueeze(0).repeat(self.n_agents, 1, 1, 1)

        # reward calc
        normalized_total_power_consumption = range_normalization(simulator_info['totoal_power_consumption'], 1, 5) # assumed range of power: 1, 5
        if self.env_args['reward'] == 'weighted_sum':
            normalized_min_sinr = range_normalization(simulator_info['min_sinr'], -75., 25.) # assumed range of min_sinr: -75, 25
            alpha = self.env_args['reward_power_consumption_coef']
            reward_ = ((1 - alpha) * normalized_min_sinr - alpha * normalized_total_power_consumption).mean()
        elif self.env_args['reward'] == 'se_requirement':
            # min_sinr - threshold is expected to be > 0
            threshold = self.env_args['sinr_threshold']
            constraints = simulator_info['min_sinr'] - threshold
            beta = self.env_args['reward_sla_viol_coef']
            if self.env_args['barrier_function'] == 'exponential':
                constraints /= 100
                se_violation_cost = torch.clip(torch.exp(-beta * constraints), max=100)
            elif self.env_args['barrier_function'] == 'step':
                se_violation_cost = beta * (constraints < 0).float()
            else:
                NotImplementedError
            reward_ = (-se_violation_cost - normalized_total_power_consumption).mean()
        else:
            raise NotImplementedError
        reward = reward_.clone().detach()
        # reward = reward_.clone().detach().unsqueeze(0).unsqueeze(0).repeat(self.n_agents, 1)

        mask = self.simulator.channel_manager.measurement_mask.clone().detach() \
            .flatten().to(torch.int32).unsqueeze(1)

        info = {
            'min_sinr': simulator_info['min_sinr'].mean(),
            'totoal_power_consumption': simulator_info['totoal_power_consumption'].mean(),
            'reward': reward_.mean(),
            'se_violation_cost': se_violation_cost.mean()
        }

        return obs, state, reward, mask, info


    def get_obs(self):
        """ Returns all agent observations in a list """
        state = self.datastore.get_last_k_elements()['obs'].permute(2, 0, 1)
        obs = state.clone().detach()

        return [item.flatten() for item in list(torch.unbind(obs, dim=0))]


    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        state = self.datastore.get_last_k_elements()['obs'].permute(2, 0, 1)
        obs = state.clone().detach()[agent_id]
        # if len(obs) < self.get_obs_size():
        #     obs = np.concatenate([obs, np.zeros((self.get_obs_size() - len(obs)))],
        #                          axis=0)  # pad all obs to same length

        return obs


    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.observation_space[0].shape[0]


    def get_state(self, team=None):
        state = self.datastore.get_last_k_elements()['obs'].permute(2, 0, 1)
        obs = state.clone().detach().flatten()

        return obs


    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.share_observation_space[0].shape[0]


    def get_avail_actions(self):
        return torch.ones((self.n_agents, self.get_total_actions()))


    def get_total_actions(self):
        return self.action_space[0].n


    def seed(self, seed):
        self.simulator.set_seed(seed)


    def reset(self):
        self.simulator.reset()
        obs, state, _, mask, info = self.compute_state_reward()

        return obs, state, mask, info


    def get_env_info(self):
        action_spaces = self.action_space
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.env_args['episode_limit'],
                    "action_spaces": action_spaces,
                    "actions_dtype": torch.int16,
                    "normalise_actions": False}

        return env_info
