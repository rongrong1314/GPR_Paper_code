'''
This library allows access to the simulated robot class, which can be designed using a number of parameters.
'''
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import scipy as sp
import math
import os
import GPy as GPy
import dubins
import time
from itertools import chain
import pdb
import logging

logger = logging.getLogger('robot')

import aq_library as aqlib
import mcts_library as mctslib
import gpmodel_library as gplib
import evaluation_library as evalib
import paths_library as pathlib
import envmodel_library as envlib
import obstacles as obslib


class Robot(object):
    ''' The Robot class, which includes the vehicles current model of the world and IPP algorithms.'''

    def __init__(self, **kwargs):
        ''' Initialize the robot class with a GP model, initial location, path sets, and prior dataset
        Inputs:
            sample_world (method) a function handle that takes a set of locations as input and returns a set of observations
            start_loc (tuple of floats) the location of the robot initially in 2-D space e.g. (0.0, 0.0, 0.0)
            extent (tuple of floats): a tuple representing the max/min of 2D rectangular domain i.e. (-10, 10, -50, 50)
            kernel_file (string) a filename specifying the location of the stored kernel values
            kernel_dataset (tuple of nparrays) a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray of type float and zvals is a Npoint x 1 nparray of type float
            prior_dataset (tuple of nparrays) a tuple (xvals, zvals), where xvals is a Npoint x 2 nparray of type float and zvals is a Npoint x 1 nparray of type float
            init_lengthscale (float) lengthscale param of kernel
            init_variance (float) variance param of kernel
            noise (float) the sensor noise parameter of kernel
            path_generator (string): one of default, dubins, or equal_dubins. Robot path parameterization.
            frontier_size (int): the number of paths in the generated path set
            horizon_length (float): the length of the paths generated by the robot
            turning_radius (float): the turning radius (in units of distance) of the robot
            sample_set (float): the step size (in units of distance) between sequential samples on a trajectory
            evaluation (Evaluation object): an evaluation object for performance metric compuation
            f_rew (string): the reward function. One of {hotspot_info, mean, info_gain, exp_info, mes}
                    create_animation (boolean): save the generate world model and trajectory to file at each timestep
        '''

        # Parameterization for the robot
        self.ranges = kwargs['extent']
        self.dimension = kwargs['dimension']
        self.create_animation = kwargs['create_animation']
        self.eval = kwargs['evaluation']
        self.loc = kwargs['start_loc']
        #self.time = kwargs['start_time']
        self.sample_world = kwargs['sample_world']
        self.f_rew = kwargs['f_rew']
        self.frontier_size = kwargs['frontier_size']
        self.discretization = kwargs['discretization']
        self.tree_type = kwargs['tree_type']
        self.path_option = kwargs['path_generator']

        self.nonmyopic = kwargs['nonmyopic']
        self.comp_budget = kwargs['computation_budget']
        self.roll_length = kwargs['rollout_length']
        self.horizon_length = kwargs['horizon_length']
        self.sample_step = kwargs['sample_step']
        self.turning_radius = kwargs['turning_radius']
        self.goal_only = kwargs['goal_only']
        self.obstacle_world = kwargs['obstacle_world']
        self.learn_params = kwargs['learn_params']
        self.use_cost = kwargs['use_cost']

        self.MIN_COLOR = kwargs['MIN_COLOR']
        self.MAX_COLOR = kwargs['MAX_COLOR']

        self.maxes = []
        self.current_max = -1000
        self.current_max_loc = [0, 0]
        self.max_locs = None
        self.max_val = None
        self.target = None
        self.noise = kwargs['noise']
        #mves方法
        self.aquisition_function = aqlib.mves

        # Initialize the robot's GP model with the initial kernel parameters
        #input:边界；尺度1；初始方差100；噪声10.0001；维度2
        #output:只赋值参数不动作
        self.GP = gplib.OnlineGPModel(ranges=self.ranges, lengthscale=kwargs['init_lengthscale'],
                                      variance=kwargs['init_variance'], noise=self.noise, dimension=self.dimension)

        # If both a kernel training dataset and a prior dataset are provided, train the kernel using both
        if kwargs['kernel_dataset'] is not None and kwargs['prior_dataset'] is not None:
            data = np.vstack([kwargs['prior_dataset'][0], kwargs['kernel_dataset'][0]])
            observations = np.vstack([kwargs['prior_dataset'][1], kwargs['kernel_dataset'][1]])
            self.GP.train_kernel(data, observations, kwargs['kernel_file'])
            # Train the kernel using the provided kernel dataset
        elif kwargs['kernel_dataset'] is not None:
            self.GP.train_kernel(kwargs['kernel_dataset'][0], kwargs['kernel_dataset'][1], kwargs['kernel_file'])
        # If a kernel file is provided, load the kernel parameters
        elif kwargs['kernel_file'] is not None:
            self.GP.load_kernel()
        # No kernel information was provided, so the kernel will be initialized with provided values
        else:
            pass

        # Incorporate the prior dataset into the model
        if kwargs['prior_dataset'] is not None:
            self.GP.add_data(kwargs['prior_dataset'][0], kwargs['prior_dataset'][1])

        # The path generation class for the robot
        #只是赋值dubins参数
        if self.path_option == 'dubins':
            self.path_generator = pathlib.Dubins_Path_Generator(self.frontier_size, self.horizon_length,
                                                                self.turning_radius, self.sample_step, self.ranges,
                                                                self.obstacle_world)
        self.visualize_world_model(screen=True, filename='FINAL')

    def choose_trajectory(self, t):
        ''' Select the best trajectory avaliable to the robot at the current pose, according to the aquisition function.
        Input:
            t (int > 0): the current planning iteration (value of a point can change with algortihm progress)
        Output:
            either None or the (best path, best path value, all paths, all values, the max_locs for some functions)
        '''
        value = {}
        param = None

        max_locs = max_vals = None
        if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            self.max_val, self.max_locs, self.target = aqlib.sample_max_vals(self.GP, t=t, visualize=True,
                                                                             f_rew=self.f_rew,
                                                                             obstacles=self.obstacle_world)
        elif self.f_rew == 'naive' or self.f_rew == 'naive_value':
            self.max_val, self.max_locs, self.target = aqlib.sample_max_vals(self.GP, t=t,
                                                                             obstacles=self.obstacle_world,
                                                                             visualize=True, f_rew=self.f_rew,
                                                                             nK=int(self.sample_num))
            param = ((self.max_val, self.max_locs, self.target), self.sample_radius)
        pred_loc, pred_val = self.predict_max(t=t)

        paths, true_paths = self.path_generator.get_path_set(self.loc)

        for path, points in paths.items():
            # set params
            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                param = (self.max_val, self.max_locs, self.target)
            elif self.f_rew == 'exp_improve':
                if len(self.maxes) == 0:
                    param = [self.current_max]
                else:
                    param = self.maxes
            #  get costs
            cost = 100.0
            if self.use_cost == True:
                cost = float(self.path_generator.path_cost(true_paths[path]))
                if cost == 0.0:
                    cost = 100.0

            # set the points over which to determine reward
            if self.path_option == 'fully_reachable_goal' and self.goal_only == True:
                poi = [(points[-1][0], points[-1][1])]
            elif self.path_option == 'fully_reachable_step' and self.goal_only == True:
                poi = [(self.goals[path][0], self.goals[path][1])]
            else:
                poi = points

            if self.use_cost == False:
                value[path] = self.aquisition_function(time=t, xvals=poi, robot_model=self.GP, param=param)
            else:
                reward = self.aquisition_function(time=t, xvals=poi, robot_model=self.GP, param=param)
                value[path] = reward / cost
        try:
            best_key = np.random.choice([key for key in value.keys() if value[key] == max(value.values())])
            return paths[best_key], true_paths[best_key], value[best_key], paths, value, self.max_locs
        except:
            return None

    def collect_observations(self, xobs):
        ''' Gather noisy samples of the environment and updates the robot's GP model.
        Input:
            xobs (float array): an nparray of floats representing observation locations, with dimension NUM_PTS x 2 '''
        zobs = self.sample_world(xobs)
        self.GP.add_data(xobs, zobs)

        for z, x in zip(zobs, xobs):
            if z[0] > self.current_max:
                self.current_max = z[0]
                self.current_max_loc = [x[0], x[1]]

    def predict_max(self, t=0):
        # If no observations have been collected, return default value
        if self.GP.xvals is None:
            return np.array([0., 0.]), 0.
        ''' Second option: generate a set of predictions from model and return max '''
        # Generate a set of observations from robot model with which to predict mean
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse=False, indexing='xy')

        if self.dimension == 2:
            data = np.vstack([x1.ravel(), x2.ravel()]).T
        elif self.dimension == 3:
            data = np.vstack([x1.ravel(), x2.ravel(), self.time * np.ones(len(x1.ravel()))]).T
        observations, var = self.GP.predict_value(data)

        if t > 50:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.set_xlim(self.ranges[0:2])
            ax2.set_ylim(self.ranges[2:])        
            ax2.set_title('Countour Plot of the Approximated World Model')     
            plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), cmap = 'viridis')
            plot = ax2.scatter(x1, x2, observations.reshape(x1.shape), cmap = 'viridis')
            plt.show()

        return data[np.argmax(observations), :], np.max(observations)

    def planner(self, T):
        ''' Gather noisy samples of the environment and updates the robot's GP model
        Input:
            T (int > 0): the length of the planning horizon (number of planning iterations)'''
        self.trajectory = []
        self.dist = 0

        for t in range(T):
            # Select the best trajectory according to the robot's aquisition function
            self.time = t
            print("[", t, "] Current Location:  ", self.loc, "Current Time:", self.time)
            logger.info("[{}] Current Location: {}".format(t, self.loc))
            #选出最佳位置
            # Let's figure out where the best point is in our world
            pred_loc, pred_val = self.predict_max()
            print("Current predicted max and value: \t", pred_loc, "\t", pred_val)
            logger.info("Current predicted max and value: {} \t {}".format(pred_loc, pred_val))

            # If myopic planner
            if self.nonmyopic == False:
                sampling_path, best_path, best_val, all_paths, all_values, max_locs = self.choose_trajectory(t=t)
            else:

                if self.f_rew == "naive" or self.f_rew == "naive_value":
                    param = (self.sample_num, self.sample_radius)
                else:
                    # set params
                    if self.f_rew == "exp_improve":
                        param = self.current_max
                    else:
                        param = None
                # create the tree search
                #input：预算代价250；GP模型；机器人被实例化的位置（50,50,0）；rollout_length = 5；dubins path；mves方程；；mes；
                #t=0;tree=dpw
                #output：对cmcts算法赋值
                mcts = mctslib.cMCTS(self.comp_budget, self.GP, self.loc, self.roll_length, self.path_generator,
                                     self.aquisition_function, self.f_rew, t, aq_param=param, use_cost=self.use_cost,
                                     tree_type=self.tree_type)
                sampling_path, best_path, best_val, all_paths, all_values, self.max_locs, self.max_val, self.target = mcts.choose_trajectory(
                    t=t)

            ''' Update eval metrics '''
            # Compute distance traveled
            start = self.loc
            for m in best_path:
                self.dist += np.sqrt((start[0] - m[0]) ** 2 + (start[1] - m[1]) ** 2)
                start = m

            # Update planner evaluation metrics
            self.eval.update_metrics(
                t=len(self.trajectory),
                robot_model=self.GP,
                all_paths=all_paths,
                selected_path=sampling_path,
                value=best_val,
                max_loc=pred_loc,
                max_val=pred_val,
                params=[self.current_max, self.current_max_loc, self.max_val, self.max_locs],
                dist=self.dist)

            if best_path == None:
                break

            # Gather data along the selected path and update GP
            data = np.array(sampling_path)
            x1 = data[:, 0]
            x2 = data[:, 1]
            if self.dimension == 2:
                xlocs = np.vstack([x1, x2]).T
            elif self.dimension == 3:
                # Collect observations at the current time
                xlocs = np.vstack([x1, x2, t * np.ones(len(x1))]).T
            else:
                raise ValueError('Only 2D or 3D worlds supported!')

            self.collect_observations(xlocs)

            # If set, learn the kernel parameters from the new data
            if t < T / 3 and self.learn_params == True:
                self.GP.train_kernel()

            self.trajectory.append(best_path)

            # If set, update the visualization
            if self.create_animation:
                self.visualize_trajectory(screen=False, filename=t, best_path=sampling_path,
                                          maxes=self.max_locs, all_paths=all_paths, all_vals=all_values)
                self.visualize_reward(screen=True, filename='REWARD.' + str(t), t=t)

            # Update the robot's current location
            self.loc = sampling_path[-1]

        np.savetxt('./figures/' + self.f_rew + '/robot_model.csv',
                   (self.GP.xvals[:, 0], self.GP.xvals[:, 1], self.GP.zvals[:, 0]))

    def visualize_trajectory(self, screen=True, filename='SUMMARY', best_path=None,
                             maxes=None, all_paths=None, all_vals=None):
        ''' Visualize the set of paths chosen by the robot
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file
            filename (string): substring for the last part of the filename i.e. '0', '1', ...
            best_path (path object)
            maxes (list of locations)
            all_paths (list of path objects)
            all_vals (list of all path rewards)
            T (string or int): string append to the figure filename
        '''

        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse=False, indexing='xy')

        if self.dimension == 2:
            data = np.vstack([x1.ravel(), x2.ravel()]).T
        elif self.dimension == 3:
            data = np.vstack([x1.ravel(), x2.ravel(), self.time * np.ones(len(x1.ravel()))]).T

        observations, var = self.GP.predict_value(data)

        # Plot the current robot model of the world
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.ranges[0:2])
        ax.set_ylim(self.ranges[2:])

        if self.MAX_COLOR is not None and self.MIN_COLOR is not None:
            plot = ax.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap='viridis', vmin=self.MIN_COLOR,
                               vmax=self.MAX_COLOR)
            if self.GP.xvals is not None:
                scatter = ax.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c='k', s=20.0, cmap='viridis',
                                     vmin=self.MIN_COLOR, vmax=self.MAX_COLOR)
        else:
            plot = ax.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap='viridis')
            if self.GP.xvals is not None:
                scatter = ax.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c='k', s=20.0, cmap='viridis')

        color = iter(plt.cm.cool(np.linspace(0, 1, len(self.trajectory))))

        # Plot the current trajectory
        '''
        if self.trajectory is not None:
            for i, path in enumerate(self.trajectory):
                c = next(color)
                f = np.array(path)
                plt.plot(f[:,0], f[:,1], c=c)

        # If available, plot the current set of options available to robot, colored
        # by their value (red: low, yellow: high)
        if all_paths is not None:
            all_vals = [x for x in all_vals.values()]   
            path_color = iter(plt.cm.autumn(np.linspace(0, max(all_vals),len(all_vals))/ max(all_vals)))        
            path_order = np.argsort(all_vals)

            for index in path_order:
                c = next(path_color)                
                points = all_paths[all_paths.keys()[index]]
                f = np.array(points)
                plt.plot(f[:,0], f[:,1], c = c)

        # If available, plot the selected path in green
        if best_path is not None:
            f = np.array(best_path)
            plt.plot(f[:,0], f[:,1], c = 'g')
        '''

        # If available, plot the current location of the maxes for mes
        if maxes is not None:
            for coord in maxes:
                plt.scatter(coord[0], coord[1], color='r', marker='*', s=500.0)
            # plt.scatter(maxes[:, 0], maxes[:, 1], color = 'r', marker = '*', s = 500.0)

        # If available, plot the obstacles in the world
        if len(self.obstacle_world.get_obstacles()) != 0:
            for o in self.obstacle_world.get_obstacles():
                x, y = o.exterior.xy
                plt.plot(x, y, 'r', linewidth=3)

        # Either plot to screen or save to file
        if screen:
            plt.show()
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig.savefig('./figures/' + str(self.f_rew) + '/trajectory-N.' + str(filename) + '.png')
            # plt.show()
            plt.close()

    def visualize_reward(self, screen=False, filename='REWARD', t=0):
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse=False, indexing='xy')  # dimension: NUM_PTS x NUM_PTS

        if self.dimension == 2:
            data = np.vstack([x1.ravel(), x2.ravel()]).T
        elif self.dimension == 3:
            data = np.vstack([x1.ravel(), x2.ravel(), self.time * np.ones(len(x1.ravel()))]).T

        print
        "Entering visualize reward"

        if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            param = (self.max_val, self.max_locs, self.target)
        elif self.f_rew == 'exp_improve':
            if len(self.maxes) == 0:
                param = [self.current_max]
            else:
                param = self.maxes
        elif self.f_rew == "naive" or self.f_rew == "naive_value":
            param = ((self.max_val, self.max_locs, self.target), self.sample_radius)
        else:
            param = None

        '''
        r = self.aquisition_function(time = t, xvals = data, robot_model = self.GP, param = param)
        print "rewrd:", r
        '''

        reward = self.aquisition_function(time=t, xvals=data, robot_model=self.GP, param=param, FVECTOR=True)

        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.set_xlim(self.ranges[0:2])
        ax2.set_ylim(self.ranges[2:])
        ax2.set_title('Reward Plot ')

        MAX_COLOR = np.percentile(reward, 98)
        MIN_COLOR = np.percentile(reward, 2)

        if MAX_COLOR > MIN_COLOR:
            # plot = ax2.contourf(x1, x2, reward.reshape(x1.shape), cmap = 'viridis', vmin = MIN_COLOR, vmax = MAX_COLOR, levels=np.linspace(MIN_COLOR, MAX_COLOR, 25))
            plot = ax2.contourf(x1, x2, reward.reshape(x1.shape), 25, cmap='plasma', vmin=MIN_COLOR, vmax=MAX_COLOR)
        else:
            plot = ax2.contourf(x1, x2, reward.reshape(x1.shape), 25, cmap='plasma')

        # If available, plot the current location of the maxes for mes
        if self.max_locs is not None:
            for coord in self.max_locs:
                plt.scatter(coord[0], coord[1], color='r', marker='*', s=500.0)

        '''
        # Plot the samples taken by the robot
        if self.GP.xvals is not None:
            scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c=self.GP.zvals.ravel(), s = 10.0, cmap = 'viridis')        
        '''
        if not os.path.exists('./figures/' + str(self.f_rew)):
            os.makedirs('./figures/' + str(self.f_rew))
        fig2.savefig('./figures/' + str(self.f_rew) + '/world_model.' + str(filename) + '.png')
        plt.close()

    def visualize_world_model(self, screen=True, filename='SUMMARY'):
        ''' Visaulize the robots current world model by sampling points uniformly in space and
        plotting the predicted function value at those locations.
        Inputs:
            screen (boolean): determines whether the figure is plotted to the screen or saved to file
            filename (String): name of the file to be made
            maxes (locations of largest points in the world)
        '''
        # Generate a set of observations from robot model with which to make contour plots
        x1vals = np.linspace(self.ranges[0], self.ranges[1], 100)
        x2vals = np.linspace(self.ranges[2], self.ranges[3], 100)
        x1, x2 = np.meshgrid(x1vals, x2vals, sparse=False, indexing='xy')  # dimension: NUM_PTS x NUM_PTS

        if self.dimension == 2:
            data = np.vstack([x1.ravel(), x2.ravel()]).T
        elif self.dimension == 3:
            data = np.vstack([x1.ravel(), x2.ravel(), self.time * np.ones(len(x1.ravel()))]).T
        observations, var = self.GP.predict_value(data)

        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.set_xlim(self.ranges[0:2])
        ax2.set_ylim(self.ranges[2:])
        ax2.set_title('Countour Plot of the Robot\'s World Model')

        if self.MAX_COLOR is not None and self.MIN_COLOR is not None:

            plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap='viridis', vmin=self.MIN_COLOR,
                                vmax=self.MAX_COLOR)
            if self.GP.xvals is not None:

                scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c='k', s=10.0)
        else:
            plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap='viridis')
            if self.GP.xvals is not None:
                scatter = ax2.scatter(self.GP.xvals[:, 0], self.GP.xvals[:, 1], c=self.GP.zvals.ravel(), s=10.0,
                                      cmap='viridis')

                # Plot the samples taken by the robot
        if screen:
            plt.show()
        else:
            if not os.path.exists('./figures/' + str(self.f_rew)):
                os.makedirs('./figures/' + str(self.f_rew))
            fig2.savefig('./figures/' + str(self.f_rew) + '/world_model.' + str(filename) + '.png')
            plt.close()

    def plot_information(self):
        ''' Visualizes the accumulation of reward and aquisition functions '''
        self.eval.plot_metrics()