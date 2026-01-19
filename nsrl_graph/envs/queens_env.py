import sys
sys.path.insert(0, '/gpfs/home4/shuaih/file2025_ns/005_supervised/004_supervised/nsrl/pypsdd/')

from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot
from itertools import combinations

import numpy as np


class QueensEnv(object):
    def __init__(self, size):
        # self.vtree = Vtree.read(vtree_dir)
        # self.manager = SddManager(self.vtree)
        
        # self.alpha = io.sdd_read(sdd_dir, self.manager)
        # self.manager, self.alpha = formula_partial_nqueens(size)
        
        self.size = size
        self.max_action_index = size*size
        self.state = None
        self.steps = 0
        self.max_steps = size * 2
        self._max_episode_steps = size * 2
        self.reset()
    
    
    def var(self, i, j):
        return self.manager.literal(i * self.size + j + 1)  
    
    def initial(self):
        n = self.size
        var_count = n * n  
        var_order = [i for i in range(1, var_count + 1)]
        vtree_type = "balanced"

        self.vtree = Vtree(var_count, var_order, vtree_type)
        self.manager = SddManager.from_vtree(self.vtree)

        def var(i, j):
            return self.manager.literal(i * n + j + 1)  

        clauses = []

        for i in range(n):
            row_vars = [var(i, j) for j in range(n)]
            clause = row_vars[0]
            for v in row_vars[1:]:
                clause |= v
            clauses.append(clause)
            for j1 in range(n):
                for j2 in range(j1 + 1, n):
                    clauses.append(~var(i, j1) | ~var(i, j2))

        for j in range(n):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    clauses.append(~var(i1, j) | ~var(i2, j))

        for d in range(-n + 1, n):
            diag = [(i, i - d) for i in range(n) if 0 <= i - d < n]
            for (i1, j1), (i2, j2) in combinations(diag, 2):
                clauses.append(~var(i1, j1) | ~var(i2, j2))

        for d in range(2 * n - 1):
            diag = [(i, d - i) for i in range(n) if 0 <= d - i < n]
            for (i1, j1), (i2, j2) in combinations(diag, 2):
                clauses.append(~var(i1, j1) | ~var(i2, j2))

        self.alpha = clauses[0]
        for clause in clauses[1:]:
            self.alpha &= clause
            
        self.pre_sat = self.alpha.model_count() > 0
        
    
    def step(self, action_index):
        i, j = self.decode_action(action_index)

        # 重复放皇后 → 立即惩罚
        if self.state[i, j] == 1:
            self.steps += 1
            if not self.pre_sat:
                performability = None
            else:
                post_sat = (self.alpha.copy() & self.var(i, j)).model_count() > 0
                performability = 1 if post_sat else 0
            return self.state.copy(), -0.1, self.steps >= self.max_steps, performability

        # 尝试放置
        next_state = self.state.copy()
        next_state[i, j] = 1
        self.alpha &= self.var(i, j)
        self.post_sat = self.alpha.model_count() > 0
        
        if not self.pre_sat:
            performability = None
        else:
            if self.post_sat:
                performability = 1
            else:
                performability = 0
        


        # performability = self.checkconstrains(self.state, next_state)

        self.state = next_state
        self.pre_sat = self.post_sat
        self.steps += 1

        # 判断是否完成（走满或已完成）
        placed = np.sum(self.state)
        conflict = (performability != 1)
        done = self.steps >= self.max_steps or (placed == self.size and not conflict)

        # 奖励策略
        if placed == self.size and not conflict:
            reward = 1
        else:
            reward = 0

        return self.state.copy(), reward, done, performability
    
    def reset(self, ):
        self.state = np.zeros((self.size, self.size), dtype=int)
        self.steps = 0
        self.initial()
        return self.state.copy()
        
    def encode_action(self, i, j):
        return i * self.size + j

    def decode_action(self, a):
        i = a // self.size
        j = a % self.size
        return i, j
    
    
    
    