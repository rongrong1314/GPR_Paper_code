from aq_library import *
import copy
import random
class cMCTS(MCTS):
    '''Class that establishes a MCTS for nonmyopic planning'''

    def __init__(self, computation_budget, belief, initial_pose, rollout_length, path_generator, aquisition_function,
                 f_rew, T, aq_param=None, use_cost=False, tree_type='dpw'):
        # Call the constructor of the super class
        super(cMCTS, self).__init__(computation_budget, belief, initial_pose, rollout_length, path_generator,
                                    aquisition_function, f_rew, T, aq_param, use_cost)
        self.tree_type = tree_type
        self.aq_param = aq_param

        # The differnt constatns use logarthmic vs polynomical exploriation
        if self.f_rew == 'mean':
            if self.tree_type == 'belief':
                self.c = 1000
            elif self.tree_type == 'dpw':
                self.c = 5000
        elif self.f_rew == 'exp_improve':
            self.c = 200
        elif self.f_rew == 'mes':
            if self.tree_type == 'belief':
                self.c = 1.0 / np.sqrt(2.0)
            elif self.tree_type == 'dpw':
                # self.c = 1.0 / np.sqrt(2.0)
                self.c = 1.0
                # self.c = 5.0
        else:
            self.c = 1.0
        print("Setting c to :", self.c)

    def choose_trajectory(self, t):
        # Main function loop which makes the tree and selects the best child
        # Output: path to take, cost of that path

        # randomly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs, self.target = sample_max_vals(self.GP, t=t, visualize=True)
            param = (self.max_val, self.max_locs, self.target)
        elif self.f_rew == 'exp_improve':
            param = [self.current_max]
        elif self.f_rew == 'naive' or self.f_rew == 'naive_value':
            self.max_val, self.max_locs, self.target = sample_max_vals(self.GP, t=t, nK=int(self.aq_param[0]),
                                                                       visualize=True, f_rew=self.f_rew)
            param = ((self.max_val, self.max_locs, self.target), self.aq_param[1])
        else:
            param = None

        # initialize tree
        if self.tree_type == 'dpw':
            self.tree = Tree(self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, t,
                             depth=self.rl, param=param, c=self.c)
        elif self.tree_type == 'belief':
            self.tree = BeliefTree(self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, t,
                                   depth=self.rl, param=param, c=self.c)
        else:
            raise ValueError('Tree type must be one of either \'dpw\' or \'belief\'')
        # self.tree.get_next_leaf()
        # print self.tree.root.children[0].children

        time_start = time.time()
        # while we still have time to compute, generate the tree
        i = 0
        while i < self.comp_budget:  # time.time() - time_start < self.comp_budget:
            i += 1
            gp = copy.copy(self.GP)
            self.tree.get_next_leaf(gp)

            if True:
                gp = copy.copy(self.GP)
        time_end = time.time()
        print("Rollouts completed in", str(time_end - time_start) + "s")

        print("Number of rollouts:", i)
        self.tree.print_tree()

        print((node.nqueries, node.reward / node.nqueries) for node in self.tree.root.children)

        # best_child = self.tree.root.children[np.argmax([node.nqueries for node in self.tree.root.children])]
        best_child = random.choice([node for node in self.tree.root.children if
                                    node.nqueries == max([n.nqueries for n in self.tree.root.children])])
        all_vals = {}
        for i, child in enumerate(self.tree.root.children):
            all_vals[i] = child.reward / float(child.nqueries)

        paths, dense_paths = self.path_generator.get_path_set(self.cp)
        return best_child.action, best_child.dense_path, best_child.reward / float(
            best_child.nqueries), paths, all_vals, self.max_locs, self.max_val, self.target