import pytest
import numpy as np
from easydict import EasyDict
from dizoo.box2d.mountaincar.envs import MountainCarEnv


@pytest.mark.envtest
class TestMountainCarEnv:

    def test_naive(self):
        env = MountainCarEnv(EasyDict({}))
        env.seed(314)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (2, )
        while True:
            random_action = env.random_action()
            timestep = env.step(random_action)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (2, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
            if timestep.done:
                break
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
