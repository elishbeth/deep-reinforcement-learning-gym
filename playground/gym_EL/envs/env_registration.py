from gym_EL.grid_world_env import *
from gym_EL.register import register

class Env2x2(GridWorldEnv):
    def __init__(self):
        super().__init__(nrows=2, ncols=2)

class Env4x3(GridWorldEnv):
    def __init__(self):
        super().__init__(nrows=4, ncols=3)


register(
    id='GridWorld-2x2-v0',
    entry_point='gym_EL.envs:Env2x2'
)

register(
    id='GridWorld-4x3-v0',
    entry_point='gym_EL.envs:Env4x3'
)
