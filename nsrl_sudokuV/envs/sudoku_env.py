import random
import numpy as np
from tensorflow.keras.datasets import mnist

hint = [np.array([[0.0,0.0],[0.0,0.0]]),
        np.array([[2,0,0],[0,0,0],[0,2,0]])/3,
        np.array([[4,0,0,0],[0,2,0,0],[0,0,0,0],[1,0,0,3]])/4,
        np.array([[0,0,5,0,1],[0,0,0,0,3],[0,1,2,0,0],[0,0,0,0,4],[0,0,0,3,0]])/5]
        
ground_truth_boards = [[np.array([[1, 2],[2, 1]])/2, np.array([[2, 1],[1, 2]])/2],
                       [np.array([[2,1,3],[1,3,2],[3,2,1]])/3, np.array([[2,3,1],[3,1,2],[1,2,3]])/3],
                       [np.array([[4,1,3,2],[3,2,1,4],[2,3,4,1],[1,4,2,3]])/4, 
                        np.array([[4,1,3,2],[3,2,4,1],[2,3,1,4],[1,4,2,3]])/4,
                        np.array([[4,3,1,2],[3,2,4,1],[2,1,3,4],[1,4,2,3]])/4],
                       [np.array([[4, 3, 5, 2, 1],[1, 2, 4, 5, 3],[3, 1, 2, 4, 5],[2, 5, 3, 1, 4],[5, 4, 1, 3, 2]])/5,
                        np.array([[4, 3, 5, 2, 1],[2, 4, 1, 5, 3],[3, 1, 2, 4, 5],[5, 2, 3, 1, 4],[1, 5, 4, 3, 2]])/5,
                        np.array([[4, 3, 5, 2, 1],[2, 5, 4, 1, 3],[3, 1, 2, 4, 5],[1, 2, 3, 5, 4],[5, 4, 1, 3, 2]])/5]]


class SudokuEnv(object):
    def __init__(self, size, numbers=None):
        self.nrows = size
        self.ncols = size
        self.ntiles = self.nrows * self.ncols
        
        self._max_episode_steps = self.ntiles * 2
        
        self.steps = 0
        
        
        self.mnist_digits = self._prepare_mnist_digits()
        
        if numbers is None:
            self.numbers = size
        else:
            self.numbers = numbers
            
        
        
        if size == 2:
            self.initial_board = hint[0]
            self.ground_truth_boards = ground_truth_boards[0]
        elif size == 3:
            self.initial_board = hint[1]
            self.ground_truth_boards = ground_truth_boards[1]
        elif size == 4:
            self.initial_board = hint[2]
            self.ground_truth_boards = ground_truth_boards[2]
        elif size == 5:
            self.initial_board = hint[3]
            self.ground_truth_boards = ground_truth_boards[3]
        else:
            self.initial_board = None
            self.ground_truth_boards = None
            
        self.state = self.init_state()
        self.max_action_index = self.ntiles * self.numbers
        


    def _prepare_mnist_digits(self):
        (x_train, y_train), _ = mnist.load_data()
        digit_dict = {i: [] for i in range(1, 10)}
        for img, label in zip(x_train, y_train):
            if 1 <= label <= 9 and len(digit_dict[label]) < 200:
                digit_dict[label].append(img)
        return digit_dict
        
        
    def to_image_state(self, state: np.ndarray) -> np.ndarray:
        blank = np.zeros((28, 28), dtype=np.uint8)
        image_tiles = np.zeros((self.nrows, self.ncols, 28, 28), dtype=np.uint8)

        for i in range(self.nrows):
            for j in range(self.ncols):
                val = state[i, j]
                if val == 0:
                    image_tiles[i, j] = blank
                else:
                    digit = int(round(val * self.numbers))
                    if digit in self.mnist_digits:
                        image_tiles[i, j] = random.choice(self.mnist_digits[digit])
                    else:
                        image_tiles[i, j] = blank

        image = image_tiles.transpose(0, 2, 1, 3).reshape(self.nrows * 28, self.ncols * 28)

        return image[np.newaxis, np.newaxis, :, :].astype(np.float32) / 255.0
        
        
    def get_avaiable_actions(self):
        avaiable_actions = np.zeros((self.nrows, self.ncols, self.numbers))
        avaiable_actions = avaiable_actions.flatten()
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.state[i][j] == 0:
                    for k in range(self.numbers):
                        action_index = self.encode_action(i+1, j+1, k+1)
                        avaiable_actions[action_index] = 1
        return avaiable_actions


    def to_one_hot(self, state: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((self.nrows, self.ncols, 9))
        for i in range(self.nrows):
            for j in range(self.ncols):
                val = state[i, j]
                if val != 0:
                    index = int(round(val * self.numbers)) - 1
                    one_hot[i, j, index] = 1
        return one_hot     
        
    def init_state(self):
        return self.initial_board.copy()
        
        
    def reset(self):
        self.steps = 0
        self.state = self.init_state()
        return self.to_image_state(self.state)
        
    def checkconstrains(self, arr: np.ndarray) -> bool:
        # print("==========")
        # print(arr)
        # print("==========")
        for gt in self.ground_truth_boards:
            if self._matches_ground_truth(arr, gt):
                return True 
        return False
        
    def _matches_ground_truth(self, state: np.ndarray, gt: np.ndarray, tol=1e-6) -> bool:
        non_zero_mask = state != 0
        return np.all(np.abs(state[non_zero_mask] - gt[non_zero_mask]) < tol)
        
    def step(self, action_index):
        # decode action
        row, col, k = self.decode_action(action_index)
        # print(action_index)
        self.steps += 1
        
        # prepare the 0.5 actions
        performability = np.ones((self.max_action_index,)) / 2
        
        
        
        if (self.state[row-1][col-1] != 0) and (self.steps <= self._max_episode_steps):
            if not self.checkconstrains(self.state):
                performability = None
            assume_state = self.state.copy()
            assume_state[row-1][col-1] = k / self.numbers
            if performability is not None:
                if self.checkconstrains(assume_state):
                    performability[action_index] = 1
                else:
                    performability[action_index] = 0
            return self.to_image_state(self.state), -0.1, False, performability
        elif self.steps > self._max_episode_steps:
            if not self.checkconstrains(self.state):
                performability = None
            assume_state = self.state.copy()
            assume_state[row-1][col-1] = k / self.numbers
            if performability is not None:
                if self.checkconstrains(assume_state):
                    performability[action_index] = 1
                else:
                    performability[action_index] = 0
            return self.to_image_state(self.state), -1, True, performability
        else:
            if not self.checkconstrains(self.state):
                performability = None
        
            # perform action
            self.state[row-1][col-1] = k / self.numbers
            
            if performability is not None:
                if self.checkconstrains(self.state):
                    performability[action_index] = 1
                else:
                    performability[action_index] = 0
                
            if np.all(self.state!=0):
                if self.is_unique_rows_cols(self.state):
                    return self.to_image_state(self.state), 1, True, performability
                else:
                    return self.to_image_state(self.state), -1, True, performability
            else:
                return self.to_image_state(self.state), 0.1, False, performability
        
    def is_unique_rows_cols(self, arr: np.ndarray) -> bool:
        n = arr.shape[0]
    
        for row in arr:
            if len(set(row)) != n:
                return False
    
        for col in arr.T:  
            if len(set(col)) != n:
                return False
    
        return True
        
        
        
    def step_old(self, action_index):
        # decode action
        row, col, k = self.decode_action(action_index)
        # print(action_index)
        self.steps += 1
        
        
        if (self.state[row-1][col-1] != 0) and (self.steps <= self._max_episode_steps):
            return self.state, -0.1, False, 0
        elif (not self.is_valid_move(row-1, col-1, k)) or (self.steps > self._max_episode_steps):
            return self.state, -1, True, 0
        else:
            self.state[row-1][col-1] = k / self.numbers
            
            if np.all(self.state!=0):
                return self.state, 1, True, 0
            else:
                return self.state, 0.1, False, 0
        
        
        
        
    def is_valid_move(self, i, j, k):
        if (k / self.numbers) in self.state[i, :] or (k / self.numbers) in self.state[:, j]:
            return False
        return True
        
    def encode_action(self, i, j, k):
        return (i - 1) * self.ncols * self.numbers + (j - 1) * self.numbers + (k - 1)
        
    def decode_action(self, A):
        i = A // (self.ncols * self.numbers) + 1
        A %= (self.ncols * self.numbers)
        j = A // self.numbers + 1
        k = A % self.numbers + 1
        return i, j, k
    
    
    # def encode_action(self, i, j, k):
    #     return (k - 1) * (self.nrows * self.ncols) + (i - 1) * self.ncols + (j - 1)

    # def decode_action(self, A):
    #     k = A // (self.nrows * self.ncols) + 1
    #     A %= (self.nrows * self.ncols) 
    #     i = A // self.ncols + 1
    #     j = A % self.ncols + 1
    #     return i, j, k


        
        
        
        