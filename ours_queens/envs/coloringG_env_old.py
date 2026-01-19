import sys
sys.path.insert(0, '/gpfs/home4/shuaih/file2025_ns/005_supervised/004_supervised/nsrl/pypsdd/')

from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot
from itertools import combinations

import numpy as np


class ColoringG(object):
    def __init__(self, graph_index = 1):
        self.graph_index = graph_index
        if graph_index == 1:
            self.node_num = 8
            self.color_num = 4
            
            self.adjacency = {
                0: [1, 2, 7, 5],
                1: [0, 2, 3, 7, 6, 4],
                2: [0, 1, 3, 4, 5],
                3: [1, 2, 4, 6],
                4: [2, 3, 5, 6, 7],
                5: [2, 4, 6, 7, 0],
                6: [3, 4, 5, 7, 1],
                7: [0, 1, 4, 5, 6],
            }
            
            self.max_action_index = self.node_num * self.color_num
            self.state = None
            self.steps = 0
            self.max_steps = self.node_num * 2
            self._max_episode_steps = self.node_num * 2
            self.reset()
            
        elif graph_index == 2:
            self.node_num = 6
            self.color_num = 3
            
            self.adjacency = {
                0: [1, 5],
                1: [0, 2, 4, 5],
                2: [1, 3],
                3: [2, 4, 1],
                4: [3, 5],
                5: [4, 0, 1],
            }
            
            self.max_action_index = self.node_num * self.color_num
            self.state = None
            self.steps = 0
            self.max_steps = self.node_num * 2
            self._max_episode_steps = self.node_num * 2
            self.reset()

        elif graph_index == 3:
            self.node_num = 8
            self.color_num = 3
            
            self.adjacency = {
                0: [1, 2, 7],
                1: [0, 3, 7],
                2: [0, 3, 5],
                3: [1, 2, 4],
                4: [3, 5, 6],
                5: [2, 4, 6],
                6: [4, 5, 7],
                7: [0, 1, 6],
            }
            
            self.max_action_index = self.node_num * self.color_num
            self.state = None
            self.steps = 0
            self.max_steps = self.node_num * 2
            self._max_episode_steps = self.node_num * 2
            self.reset()
            
        elif graph_index == 4:
            self.node_num = 8
            self.color_num = 3
            
            self.adjacency = {
                0: [1, 2, 7],
                1: [0, 3, 7],
                2: [0, 3, 5],
                3: [1, 2, 4],
                4: [3, 5, 6],
                5: [2, 4, 6],
                6: [4, 5, 7],
                7: [0, 1, 6],
            }
            
            self.max_action_index = self.node_num * self.color_num
            self.state = None
            self.steps = 0
            self.max_steps = self.node_num * 2
            self._max_episode_steps = self.node_num * 2
            self.reset()
        
        
        
    def initial(self):
        num_nodes = self.node_num
        num_colors = self.color_num
        var_count = num_nodes * num_colors
        var_order = [i for i in range(1, var_count + 1)]
        vtree_type = "balanced"

        self.vtree = Vtree(var_count, var_order, vtree_type)
        self.manager = SddManager.from_vtree(self.vtree)

        def var(node, color):
            """Return the literal that node uses this color"""
            return self.manager.literal(node * num_colors + color + 1)
        
        self.var = var
        
        clauses = []

        # 每个节点至少有一个颜色
        for node in range(num_nodes):
            clause = var(node, 0)
            for c in range(1, num_colors):
                clause |= var(node, c)
            clauses.append(clause)

        # 每个节点至多一个颜色（任意两个颜色不能同时选）
        for node in range(num_nodes):
            for c1, c2 in combinations(range(num_colors), 2):
                clauses.append(~var(node, c1) | ~var(node, c2))

        # 邻接节点不能同色
        added = set()
        for node in range(num_nodes):
            for neighbor in self.adjacency[node]:
                if (neighbor, node) in added:  # skip symmetric pair
                    continue
                for c in range(num_colors):
                    clauses.append(~var(node, c) | ~var(neighbor, c))
                added.add((node, neighbor))

        # AND 所有约束成 α
        self.alpha = clauses[0]
        for clause in clauses[1:]:
            self.alpha &= clause
    
        if self.graph_index == 1 or self.graph_index == 3:
            self.alpha &= var(0, 0)
            self.alpha &= var(1, 1)
        elif self.graph_index == 2:
            pass
        elif self.graph_index == 4:
            self.alpha &= var(0, 0)
        
        self.pre_sat = self.alpha.model_count() > 0
        
        # print("Model counting: {}".format(alpha.model_count()))
        
        
    def reset(self, ):
        if self.graph_index == 1 or self.graph_index == 3:
            list_color = [1,2,0,0,0,0,0,0]
        elif self.graph_index == 2:
            list_color = [0,0,0,0,0,0]
        elif self.graph_index == 4:
            list_color = [1,0,0,0,0,0]
            
        self.state = np.array(list_color, dtype=int)  
        self.steps = 0
        self.initial()
        return self.state_to_onehot(self.state.copy())
        
    def state_to_onehot(self, state):
        # 返回 (node_num, color_num) 的 one-hot 编码矩阵
        onehot = np.zeros((self.node_num, self.color_num+1), dtype=int)
        for i, c in enumerate(state):
            if c >= 0 and c <= self.color_num:
                onehot[i, c] = 1
        return onehot
        
    def step(self, action_index):
        node, color = self.decode_action(action_index)

        # 如果该节点已经有颜色，惩罚
        if self.state[node] != 0:
            self.steps += 1
            if not self.pre_sat:
                performability = None
            else:
                post_sat = (self.alpha.copy() & self.var(node, color)).model_count() > 0
                performability = 1 if post_sat else 0
            return self.state_to_onehot(self.state.copy()), -0.1, self.steps >= self.max_steps, performability

        # 尝试上色
        next_state = self.state.copy()
        next_state[node] = color + 1

        # 更新SDD
        self.alpha &= self.var(node, color)
        self.post_sat = self.alpha.model_count() > 0

        # 判断是否合法（performability）
        if not self.pre_sat:
            performability = None
        else:
            performability = 1 if self.post_sat else 0

        # 更新状态
        self.state = next_state
        self.pre_sat = self.post_sat
        self.steps += 1

        # 检查是否完成
        done = self.steps >= self.max_steps or (np.all(self.state > 0) and performability == 1)

        # 奖励策略
        reward = 1 if (np.all(self.state > 0) and performability == 1) else 0

        return self.state_to_onehot(self.state.copy()), reward, done, performability
    
        
    def encode_action(self, node, color):
        return node * self.color_num + color

    def decode_action(self, action_index):
        node = action_index // self.color_num
        color = action_index % self.color_num
        return node, color
    
        
        
        
        
        
        