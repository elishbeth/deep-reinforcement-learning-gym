import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random as r
import time
import pyglet

""" Author: Elisabeth Landgren creds to Charles Zhang for creating a similar simulation 
    Date: June 16th
    Last Edited: June 22nd
"""

################################################
#######this is used for rendering text##########
class DrawText:
    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self):
        self.label.draw()
################################################


class GridWorldEnv(gym.Env):
    """
    Description:
        This environment is a simple nxm grid. The goal is to traverse every
        square in the grid without revisiting any square.

    Observation:
        Type: Discrete(nrows*ncols)
            i,j = (i * ncols) + j

            Example:
                   0        | 1        | 2
                ||==========|==========|==============
              0 || 0        | 1        | 2
                ||__________|__________|______________
              1 || 3        | 4        | 5
                ||__________|__________|______________
              2 || 6        | 7        | 8

    Actions:
        Type: Discrete(4)
        Num   Action
        0     Up
        1     Down
        2     Right
        3     Left

        All of these actions are a movement by one square

    Reward:
        Reward is:
        + for visiting a new square
        - for revisiting an old square
        + for visiting every square

    Starting State:
        Starting state is (0,0) with (0,0) grid set to 1 visit and all other grids set to 0 visits

    Episode Termination:
        All grids have been visited once or step count exceeds max steps


    """

    metadata = {'render.modes': ['human']}

    def __init__(self, nrows: int=2, ncols: int=2, initial: tuple = (0, 0), max_steps: int = 1000):
        super(GridWorldEnv, self).__init__()

        self.nrows = nrows
        self.ncols = ncols
        self.initial = initial
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Discrete(self.nrows * self.ncols)
        self.seed()
        self.state = None
        self.current_step = None
        self.board = None
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _take_action(self, action: int):

        ''' This function takes an action int and performs that action, moving the agents state by one
         grid in the corresponding direction as long as the move is within the bounds of the board. If the action
         would got off the board the agent remains put. It then increases the number of visits to that new state
          by adding one to that position on the board

        0     Up
        1     Down
        2     Right
        3     Left
        '''

        if action == 0:  # up
            next_state = (self.state[0]+1, self.state[1])
        elif action == 1:  # down
            next_state = (self.state[0]-1, self.state[1])
        elif action == 2:  # right
            next_state = (self.state[0], self.state[1]+1)
        elif action == 3:  # left
            next_state = (self.state[0], self.state[1]-1)
        else:
            next_state = None
        if next_state is not None and (next_state[0] >= 0) and (next_state[0] < self.nrows):
            if (next_state[1] >= 0) and (next_state[1] < self.ncols):
                self.state = next_state
        self.board[self.state] += 1  # add a visit to this location

    def _is_end(self):
        ''' This function checks if the episode is over by checking if all the grid squares have been
         visited and returns a boolean '''
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.board[i, j] == 0:
                    return False  # if any grid has not been visited, not done
        return True

    def _determine_reward(self, done):
        ''' This function determines the reward for each 'type' of move. These numbers were determined arbitrarily '''
        if done:  # done is the only exit
            return -5
        else:
            # return -0.01  # going anywhere will reduce reward.. better hurry up agent!
            return 1

    def _next_observation(self):
        ''' This function calulates the N of the grid coordinate i,j and returns it.
        N_(i,j) = i*ncols + j
        '''

        i, j = self.state

        return(i * self.ncols) + j

    def step(self, action: int):
        ''' This function performs the action, increases the number of steps taken, and determines a reward for the step '''


        # if done:
        #     self.reset()
        #     obs = self._next_observation()
        #     reward = None
        #     return obs, reward, done, {}

        self._take_action(action)
        done = self._is_end()
        reward = self._determine_reward(done)
        obs = self._next_observation()

        self.current_step += 1
        if self.current_step > self.max_steps:
            # self.reset()
            done = True

        return obs, reward, done, {}

    def reset(self):
        ''' This function resets the board to the start position. This needs to be called to initialize the board. It gives an intial observation'''
        self.board = np.zeros([self.nrows, self.ncols])
        self.board[self.initial] += 1
        self.state = self.initial
        self.current_step = 0

        obs = self._next_observation()


        return obs



    def render(self, mode='human', close=False):
        ''' This function renders the board at any given step and shows the location of the agent with a black dot.
        Each square can be White: Not visited, Green: Visited exactly once, Red: Visited more than once '''


        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        u_size = 40
        width = u_size * self.ncols
        height = u_size * self.nrows
        gap = 2  # gap between grids

        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

            # draws grids
            for n in range(self.ncols+1):
                line = rendering.Line(start=(n*u_size, 0),
                                      end=(n*u_size, u_size*self.nrows))
                self.viewer.add_geom(line)
            for n in range(self.nrows):
                line = rendering.Line(start=(0, n*u_size),
                                      end=(u_size*self.ncols, n*u_size))
                self.viewer.add_geom(line)


        for i in range(self.nrows):
            for j in range(self.ncols):
                v = [(j * u_size + gap, i * u_size + gap),
                     ((j + 1) * u_size - gap, i * u_size + gap),
                     ((j + 1) * u_size - gap, (i + 1) * u_size - gap),
                     (j * u_size + gap, (i + 1) * u_size - gap)]
                # v = [(j * u_size + gap, (self.nrows*u_size +gap) -(i * u_size + gap)),
                #      ((j + 1) * u_size - gap,(self.nrows*u_size +gap) - (i * u_size + gap)),
                #      ((j + 1) * u_size - gap, (self.nrows*u_size +gap) -((i + 1) * u_size - gap)),
                #      (j * u_size + gap, (self.nrows*u_size +gap) -((i + 1) * u_size - gap))]

                rect = rendering.FilledPolygon(v)

                if self.board[i, j] == 0:
                    rect.set_color(1.0, 1.0, 1.0)
                    self.viewer.add_geom(rect)

                if self.board[i, j] == 1:
                    rect.set_color(0, 0.8, 0.4)
                    self.viewer.add_geom(rect)

                if self.board[i, j] > 1:
                    rect.set_color(0.8, 0.4, 0.4)
                    self.viewer.add_geom(rect)

                text = str(int(self.board[i,j]))
                label = pyglet.text.Label(text, font_size=20,
                                          x=20+40*j, y=20+40*i, anchor_x='center', anchor_y='center',
                                          color=(0, 0, 0, 255))
                label.draw()
                self.viewer.add_geom(DrawText(label))


        # agent
        self.agent = rendering.make_circle(radius=u_size / 3, res=30, filled=False)
        self.agent.set_linewidth(5)
        self.viewer.add_geom(self.agent)
        self.agent_trans = rendering.Transform()
        self.agent.add_attr(self.agent_trans)

        x, y = self.state
        self.agent_trans.set_translation((y + 0.5) * u_size, (x + 0.5) * u_size)

        #I am not sure what the difference is between these two:

        # self.viewer.render(return_rgb_array=mode == 'rgb_array')
        self.viewer.render(return_rgb_array=False)

    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__ == "__main__":

    env = GridWorldEnv(4,3)
    env.reset()
    actions = range(env.action_space.n)
    for i in range(env.max_steps):
        choice = r.choice(actions)
        env.step(choice)
        env.render()
        time.sleep(.2)
    env.close()



