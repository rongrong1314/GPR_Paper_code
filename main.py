#standard imports
import os
import time
import sys
import logging
import numpy as np
import robot_library as roblib
#自写库
#更新机器人理解环境库
import envmodel_library as envlib
#创建地图库
import obstacles as obslib
#评估库
import evaluation_library as evalib

#一些flag参数
SEED =  0
REWARD_FUNCTION = 'mes' #robot will use Maximum Value Information as value function
PATHSET = 'dubins' #robot will have a set of pre-defined action sets to use for each step of planning
USE_COST = False #since all of the path options will be nominally the same length anyway in the dubins setting, robot doesn't consider path length cost
NONMYOPIC = True #robot will consider future outcomes for each action
GOAL_ONLY = False #robot will consider the whole trajectory with respect to reward
TREE_TYPE = 'dpw' #robot will use progressive widening in the search tree

# Parameters for plotting based on the seed world information
MIN_COLOR = -25.
MAX_COLOR = 25.

# 设置记录数据的路径
if not os.path.exists('./figures/' + str(REWARD_FUNCTION)):
    os.makedirs('./figures/' + str(REWARD_FUNCTION))
logging.basicConfig(filename = './figures/'+ REWARD_FUNCTION + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')

#设置world的范围
ranges = (0., 10., 0., 10.)

#加载无障碍物环境
ow = obslib.FreeWorld() #a world without obstacles

#创建机器人能够识别的环境
#没有观测值的预设先验的环境更新
world = envlib.Environment(ranges = ranges,
                           NUM_PTS = 20,
                           variance = 100.0,
                           lengthscale = 1.0,
                           visualize = False,
                           seed = SEED,
                           MIN_COLOR=MIN_COLOR,
                           MAX_COLOR=MAX_COLOR,
                           obstacle_world = ow,
                           noise=10.0)
#将更新好后的地图和奖励函数类型
# Create the evaluation class used to quantify the simulation metrics
#只初始化参数没有动作
evaluation = evalib.Evaluation(world = world, reward_function = REWARD_FUNCTION)
#设置机器人参数

robot = roblib.Robot(sample_world = world.sample_value, #function handle for collecting observations
                     start_loc = (5.0, 5.0, 0.0), #where robot is instantiated
                     dimension = 2,
                     extent = ranges, #extent of the explorable environment
                     kernel_file = None,
                     kernel_dataset = None,
                     #prior_dataset =  (data, observations),
                     prior_dataset = None,
                     init_lengthscale = 1.0,
                     init_variance = 100.0,
                     noise = 10.0001,
                     #noise = 0.5000,
                     path_generator = PATHSET, #options: default, dubins, equal_dubins, fully_reachable_goal, fully_reachable_step
                     goal_only = GOAL_ONLY, #select only if using fully reachable step and you want the reward of the step to only be the goal
                     frontier_size = 15,
                     horizon_length = 1.5,
                     turning_radius = 0.05,
                     sample_step = 0.5,
                     evaluation = evaluation,
                     f_rew = REWARD_FUNCTION,
                     create_animation = True, #logs images to the file folder
                     learn_params = False, #if kernel params should be trained online
                     nonmyopic = NONMYOPIC,
                     discretization = (20, 20), #parameterizes the fully reachable sets
                     use_cost = USE_COST, #select if you want to use a cost heuristic
                     MIN_COLOR = MIN_COLOR,
                     MAX_COLOR = MAX_COLOR,
                     computation_budget = 250.0,
                     rollout_length = 5,
                     obstacle_world = ow,
                     tree_type = TREE_TYPE)

# Establisht the number of "steps" or "actions" the robot should take in the total mission using T
#路径点
robot.planner(T = 5) #robot will plan 5 actions
robot.visualize_trajectory(screen = True) #creates a summary trajectory image
# robot.plot_information() #plots all of the metrics of interest