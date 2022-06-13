from typing import Union, Any, List, Callable, Dict, Optional
from collections import namedtuple
import random
import torch
import treetensor.numpy as tnp
from easydict import EasyDict
from unittest.mock import Mock
from copy import deepcopy

from ding.torch_utils import to_device
from ding.league.player import PlayerMeta
from ding.league.v2 import BaseLeague, Job
from ding.framework.storage import FileStorage
from ding.policy import PPOPolicy
from dizoo.distar.envs.distar_env import DIStarEnv


obs_dim = [2, 2]
action_space = 1
env_num = 2

CONFIG = dict(
    seed=0,
    policy=dict(
        learn=dict(
            update_per_collect=4,
            batch_size=8,
            learner=dict(hook=dict(log_show_after_iter=10), ),
        ),
        collect=dict(
            n_sample=16,
            unroll_len=1,
            n_episode=16,
        ),
        eval=dict(evaluator=dict(eval_freq=10), ),
        other=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=10000,
        ), ),
    ),
    env=dict(
        n_evaluator_episode=5,
        stop_value=2.0,
    ),
)
CONFIG = EasyDict(CONFIG)


class MockPolicy(Mock):

    def __init__(self) -> None:
        super(MockPolicy, self).__init__()
        self.action_space = action_space
        self.obs_dim = obs_dim

    def reset(self, data_id: Optional[List[int]] = None) -> None:
        return

    def forward(self, data: dict, **kwargs) -> dict:
        res = {}
        for i, v in data.items():
            res[i] = {'action': torch.sum(v)}
        return res

    def process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': torch.rand(self.obs_dim),
            'next_obs': torch.rand(self.obs_dim),
            'action': torch.zeros(self.action_space),
            'logit': 1.0,
            'value': 2.0,
            'reward': 0.1,
            'done': True,
        }
        return transition


class MockEnv(Mock):

    def __init__(self) -> None:
        super(MockEnv, self).__init__()
        self.env_num = env_num
        self.obs_dim = obs_dim
        self.closed = False
        self._reward_grow_indicator = 1

    @property
    def ready_obs(self) -> tnp.array:
        return tnp.stack([
            torch.zeros(self.obs_dim),
            torch.ones(self.obs_dim),
        ])

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        return

    def launch(self, reset_param: Optional[Dict] = None) -> None:
        return

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        return

    def step(self, actions: tnp.ndarray) -> List[tnp.ndarray]:
        timesteps = []
        for i in range(self.env_num):
            timestep = dict(
                obs=torch.rand(self.obs_dim),
                reward=1.0,
                done=True,
                info={'final_eval_reward': self._reward_grow_indicator * 1.0},
                env_id=i,
            )
            timesteps.append(tnp.array(timestep))
        self._reward_grow_indicator += 1  # final_eval_reward will increase as step method is called
        return timesteps


class MockHerRewardModel(Mock):

    def __init__(self) -> None:
        super(MockHerRewardModel, self).__init__()
        self.episode_size = 8
        self.episode_element_size = 4

    def estimate(self, episode: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [[episode[0] for _ in range(self.episode_element_size)]]


class MockLeague(BaseLeague):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.update_payoff_cnt = 0
        self.update_active_player_cnt = 0
        self.create_historical_player_cnt = 0
        self.get_job_info_cnt = 0

    def update_payoff(self, job):
        self.update_payoff_cnt += 1

    def update_active_player(self, meta):
        self.update_active_player_cnt += 1

    def create_historical_player(self, meta):
        self.create_historical_player_cnt += 1

    def get_job_info(self, player_id):
        self.get_job_info_cnt += 1
        other_players = [i for i in self.active_players_ids if i != player_id]
        another_palyer = random.choice(other_players)
        return Job(
            launch_player=player_id,
            players=[
                PlayerMeta(player_id=player_id, checkpoint=FileStorage(path=None), total_agent_step=0),
                PlayerMeta(player_id=another_palyer, checkpoint=FileStorage(path=None), total_agent_step=0)
            ]
        )


class MockLogger():

    def add_scalar(*args):
        pass

    def close(*args):
        pass

    def flush(*args):
        pass


class DIStarMockPolicy(PPOPolicy):
    def _mock_data(self, data):
        print("Data: ", data)
        mock_data = {}
        for id, val in data.items():
            assert isinstance(val, dict)
            assert 'obs' in val
            assert 'next_obs' in val
            assert 'action' in val
            assert 'reward' in val
            mock_data[id] = deepcopy(val)
            mock_data[id]['obs'] = torch.rand(16)
            mock_data[id]['next_obs'] = torch.rand(16)
            mock_data[id]['action'] = torch.rand(4)
        return mock_data, data

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        print("Call forward_learn:")
        mock_data, original_data = self._mock_data(data)
        return super()._forward_learn(mock_data)

    def _forward_collect(self, data: Dict[int, Any]) -> Dict[int, Any]:
        print("Call forward_collect:")
        mock_data, original_data = self._mock_data(data)
        output = super()._forward_collect(mock_data)
        for id in output.keys():
            output[id]['action'] = DIStarEnv.random_action(original_data[id]['obs'])
        return output

    # def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
    #     data = {i: torch.rand(self.policy.model.obs_shape) for i in range(self.cfg.env.collector_env_num)}
    #     return super()._forward_eval(data)
