import numpy as np
from coloringG_env import ColoringG 
from itertools import product


def play_sudoku():
    env = ColoringG(1)
    
    print("=== Welcome to QueensEnv ===")
    print("Input format: row column (starting from 0), for example `2 3`")
    print("Type `exit` to quit the game")

    while True:
        
        # print("\navaiable actions:")
        # print(env.get_avaiable_actions())
        print("\n current board:")
        print(env.state)  # Show the visible board to the player

        # User input
        move = input("Enter coordinates (row column): ").strip()
        if move.lower() == "exit":
            print("Game over!")
            break

        try:
            row, col = map(int, move.split())
        except ValueError:
            print("Invalid input format, please enter two integers, for example `2 3`")
            continue

        index = env.encode_action(row, col)
        print("===================================================")
        print("\n board before the action:")
        print(env.state)  # Show the visible board to the player
        print(f"flatten state is {env.state.flatten()}")
        print(f"action index: {index}")
        print(f"decoded action: {env.decode_action(index)}")
        
        # Execute the dig action
        state, reward, done, performa = env.step(index)

        # Show result
        print(f"fill at ({row}, {col}), reward: {reward}")
        
        print("\n board after the action:")
        print(env.state)  # Show the visible board to the player
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