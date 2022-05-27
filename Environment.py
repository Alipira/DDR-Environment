from math import pi, atan2, cos, sin
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
from DDR import DifferentialDriveRobot
 
class Env_PNG(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, t0, tf, dt, robot_init, goal:np.array, show_animation=True):
        self.i = 0
        self.t0 = t0
        self.dt = dt
        self.tf = tf
        self.t = np.arange(start=self.t0, stop=self.tf, step=dt)
        self.simulation_time = self.tf
        self.show_animation = show_animation
        self.tavg_reward = np.zeros((len(self.t)+1, 1)) # total average reward
        self.treward = np.zeros((len(self.t)+1, 1))    # total cumulative reward
        self.time_step = np.zeros((len(self.t)+1, 1))


        # Defining robot parameters
        self.trobot_pose = np.zeros((len(self.t)+1,3)) # Total positions of robot
        self.trobot_pose[0] = robot_init[0, :]
        self.Robot = DifferentialDriveRobot(self.trobot_pose[0])

        # Defining Target parameters
        self.goal = goal

        self.fig = plt.figure('Results') 
        self.ax1 = plt.subplot2grid(shape=(4, 3), loc=(0, 0), colspan=2, rowspan=2) # map
        self.ax2 = plt.subplot2grid(shape=(4, 3), loc=(2, 0), colspan=3) # Reward
        self.ax3 = plt.subplot2grid(shape=(4, 3), loc=(3, 0), colspan=3) # Time step
        self.ax4 = plt.subplot2grid(shape=(4, 3), loc=(0, 2), colspan=1) # robot's linear velocity
        self.ax5 = plt.subplot2grid(shape=(4, 3), loc=(1, 2), colspan=1) # robot's angular velocity

        # defining action space
        low = np.array([0, -10]) # shape = (2,)
        high = np.array([5, 10])
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        # Observation Space
        self.observation_space = spaces.Box(low=-30, high=30, shape=(7,), dtype=np.float32)
        self.seed()
        self.viewer = None

    # Definig Auxilary function
    def get_observation(self, action, counter):

        robot_state = self.trobot_pose[counter, :]
        goal_state = self.goal # as goal is stationary
        v, w = action # FIXME:
        xr = (goal_state[0, 0] - robot_state[0, 0]) # Relative distance along x-axis
        yr = (goal_state[0, 1] - robot_state[0, 1]) # Relative distance along y-axis
        return np.concatenate((robot_state, xr, yr, v, w), axis=None)

    def compute_reward(self, observation, simulation_time):

        reward_time_delay = -1
        reach = 0
        loose = 0
        collision = 0 # TODO: fix this in obstacles
        relative_distance = np.sqrt(observation[3]**2 + observation[4]**2)

        if relative_distance <= 0.25:
            reach = 900 
        elif simulation_time <= 0:
            loose = -100
        
        Reward = reward_time_delay + reach + loose + collision
        return Reward
    
    def is_done(self, observation, simulation_time):

        relative_distance = np.sqrt(observation[3]**2 + observation[4]**2)
        done = False
        imax = len(self.t)-1

        if relative_distance <= 0.25 or simulation_time <= 0 or self.i >= imax:
            done = True
        return done

    def plot_arrow(self, x, y, yaw, length=1.5, width=1, fc="r", ec="k"):
        if not isinstance(x, float):
            for ix, iy, iyaw in zip(x, y, yaw):
                self.plot_arrow(ix, iy, iyaw)
        else:
            self.ax1.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
            self.ax1.plot(x, y)

    # Main functions
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        action = action[0] # TODO: check the action shape
        robot_pose, _, _ = self.Robot.kinematic(action)
        self.trobot_pose[self.i+1] = robot_pose

        # Getting observation
        observation = self.get_observation(action, self.i)

        # Reduce time 
        self.simulation_time -= self.dt

        # Check if is done 
        done = self.is_done(observation,self.simulation_time)
        reward = self.compute_reward(observation, self.simulation_time)
        self.i += 1
        return observation, reward, done, {}

    def reset(self):
        
        self.simulation_time = self.tf
        self.i = 0
        v = 0
        w = 0
        self.goal = self.get_state(self.goal)
        self.trobot_pose = np.zeros((len(self.t)+1,3))
        self.trobot_pose = self.Robot.rest_pose() # Robot angel is 0 radian
        relative_distance = np.linalg.norm(self.goal[0, :2] - self.trobot_pose[0, :2])
        observation = np.concatenate((self.trobot_pose, relative_distance, v, w), axis=None)
        return observation

    def render(self): # FIXME:
    
        robot_pose = self.trobot_pose[self.i, :]
        if self.show_animation: 
            self.ax1.cla()
            self.ax2.cla()

            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            self.plot_arrow(robot_pose[self.i, 0], robot_pose[self.i, 1], robot_pose[self.i, 2])

            self.ax1.plot(self.trobot_pose[0:self.i, 0], self.trobot_pose[0:self.i, 1], "--b", label="Robot")
            self.ax1.plot(self.goal[0, 0], self.goal[0, 1], "rx", label="Goal Point")
            self.ax1.set_title("MAP")
            self.ax1.axis([-15, 20, -15, 20])
            self.ax1.grid(True)
            self.ax2.plot(self.t[0:self.i], total_w_com[0:self.i], linewidth=2, color='C1')
            self.ax2.set_xlim(left=-1, right=100, emit=True)
            self.ax2.grid(True)
            self.ax1.legend(loc=4)   # 4 is equal to lower right
            plt.tight_layout()
            plt.pause(self.dt)
        
        return np.array([[[1, 1, 1]]], dtype=np.uint8)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
