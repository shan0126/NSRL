import numpy as np
from sudoku_env import SudokuEnv 
from itertools import product

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
    

def play_sudoku():
    env = SudokuEnv(2)
    
    print("=== Welcome to Sudoku ===")
    print("Input format: row column (starting from 0), for example `2 3`")
    print("Type `exit` to quit the game")

    while True:
        
        # print("\navaiable actions:")
        # print(env.get_avaiable_actions())
        print("\n current board:")
        print_board(env.state)  # Show the visible board to the player

        # User input
        move = input("Enter coordinates (row column): ").strip()
        if move.lower() == "exit":
            print("Game over!")
            break

        try:
            row, col, k = map(int, move.split())
        except ValueError:
            print("Invalid input format, please enter two integers, for example `2 3 3`")
            continue

        index = env.encode_action(row, col, k)
        print("===================================================")
        print("\n board before the action:")
        print_board(env.state)  # Show the visible board to the player
        print(f"flatten state is {env.state.flatten()}")
        print(f"action index: {index}")
        print(f"decoded action: {env.decode_action(index)}")
        
        # Execute the dig action
        state, reward, done, performa = env.step(index)

        # Show result
        print(f"fill at ({row}, {col}) with number {k}, reward: {reward}")
        
        print("\n board after the action:")
        print_board(env.state)  # Show the visible board to the player
        print(f"flatten state is {env.state.flatten()}")
        
        print(f"the performa is {performa}")

        if done:
            if reward > 0:
                print("\n?? Congratulations, you won!")
            else:
                print("\n?? You hit a mine, game over!")
                
            env.reset()
            # print("Final board:")
            # print_board(env.board)  # Show the full board
            # break
        print("===================================================")
            
def print_board(state):
    """Format and print the board."""
    rows = []
    for row in range(state.shape[0]):  # Assuming the board is 5x5 based on the provided output
        row_values = []
        for col in range(state.shape[1]):
            row_values.append(state[row][col])
        row_values = [str(int(rv*2)) for rv in row_values]
        rows.append(" ".join(row_values))
    
    # Print the formatted rows of the board
    for row in rows:
        print(row)

    
if __name__ == "__main__":
    # trainset, labelset = sudoku2_partial()
    # print(trainset[:10])
    # print(labelset[:10])
    # exit(0)
    play_sudoku()
