#standard imports
import os
import time
import sys
import logging
import numpy as np
#自写库
import envmodel_library as envlib
import obstacles as obslib

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
world = envlib.Environment(ranges = ranges,
                           NUM_PTS = 20,
                           variance = 100.0,
                           lengthscale = 1.0,
                           visualize = True,
                           seed = SEED,
                           MIN_COLOR=MIN_COLOR,
                           MAX_COLOR=MAX_COLOR,
                           obstacle_world = ow,
                           noise=10.0)