import numpy as np

from itertools import product

solve = [[1, 2, 2, 1], [2, 1, 1, 2]]
structure_f = [[1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]]

def sudoku2_dataset():
    x = np.array([[1, 2, 2, 1], [2, 1, 1, 2]])  # 2x4
    y = np.array([[1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]])  # 2x8
    
    masks = np.array(list(product([0, 1], repeat=4)))  # 16x4
    
    expanded_x = np.vstack([x[i] * masks for i in range(2)])  # 32x4
    expanded_y = np.repeat(y, repeats=len(masks), axis=0)  # 32x8
    
    expanded_x = expanded_x/2
    
    zero_indices = np.where(np.all(expanded_x == 0, axis=1))[0]
    
    if len(zero_indices) > 0:
        first_zero_index = zero_indices[0]
        expanded_y[first_zero_index] = 1 
        
        mask = np.ones(len(expanded_x), dtype=bool)
        mask[zero_indices[1:]] = False 
        
        expanded_x = expanded_x[mask]
        expanded_y = expanded_y[mask]
    

    return expanded_x, expanded_y


def sudoku2_partial():
    x = np.array([[1, 2, 2, 1], [2, 1, 1, 2]])  # 2x4
    y = np.array([[1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]])  # 2x8
    
    masks = np.array(list(product([0, 1], repeat=4)))  # 16x4
    
    expanded_x = np.vstack([x[i] * masks for i in range(2)])  # 32x4
    expanded_x = expanded_x / 2  
    
    expanded_x = np.repeat(expanded_x, repeats=8, axis=0)  # 256x4
    expanded_y = np.repeat(y, repeats=16, axis=0)  # 32x8
    expanded_y = np.repeat(expanded_y, repeats=8, axis=0)  # 256x8
    
    y_mask = np.tile(np.eye(8), (len(expanded_y) // 8, 1))  # 256x8
    expanded_y = expanded_y * y_mask + (1 - y_mask) * 0.5
    
    return expanded_x, expanded_y
    