from copy import deepcopy
import pytest
from ding.envs import BaseEnvManager, EnvSupervisor
from ding.framework.context import BattleContext
from ding.framework.middleware import StepLeagueActor, LeagueCoordinator
from ding.framework.supervisor import ChildType

from ding.model import VAC
from ding.framework.task import task
from ding.framework.middleware.tests import cfg, MockLeague, MockLogger
from dizoo.distar.envs.distar_env import DIStarEnv
from ding.framework.middleware.tests import DIStarMockPolicy, DIStarMockPolicyCollect
from ding.framework.middleware.functional.collector import battle_inferencer_for_distar, battle_rolloutor_for_distar
from distar.ctools.utils import read_config
from unittest.mock import patch
import os

cfg = deepcopy(cfg)
env_cfg = read_config('./test_distar_config.yaml')


class PrepareTest():

    @classmethod
    def get_env_fn(cls):
        return DIStarEnv(env_cfg)

    @classmethod
    def get_env_supervisor(cls):
        env = EnvSupervisor(
            type_=ChildType.THREAD,
            env_fn=[cls.get_env_fn for _ in range(cfg.env.collector_env_num)],
            **cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    @classmethod
    def policy_fn(cls):
        model = VAC(**cfg.policy.model)
        policy = DIStarMockPolicy(cfg.policy, model=model)
        return policy

    @classmethod
    def collect_policy_fn(cls):
        policy = DIStarMockPolicyCollect()
        return policy


def _main():
    league = MockLeague(cfg.policy.other.league)
    n_players = len(league.active_players_ids)
    print(n_players)

    with task.start(async_mode=True, ctx=BattleContext()):
        with patch("ding.framework.middleware.collector.battle_inferencer", battle_inferencer_for_distar):
            with patch("ding.framework.middleware.collector.battle_rolloutor", battle_rolloutor_for_distar):
                player_0 = league.active_players[0]
                learner_0 = LeagueLearner(cfg, PrepareTest.policy_fn, player_0)
                learner_0._learner._tb_logger = MockLogger()

                player_1 = league.active_players[1]
                learner_1 = LeagueLearner(cfg, PrepareTest.policy_fn, player_1)
                learner_1._learner._tb_logger = MockLogger()

                task.use(LeagueCoordinator(league))
                task.use(StepLeagueActor(cfg, PrepareTest.get_env_supervisor, PrepareTest.collect_policy_fn))
                task.use(learner_0)
                task.use(learner_1)

                task.run(max_step=300)


@pytest.mark.unittest
def test_league_actor():
    _main()


if __name__ == '__main__':
    _main()