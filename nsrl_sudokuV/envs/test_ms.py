import numpy as np
from minesweeper_env import MinesweeperEnv

def play_minesweeper():
    width, height, n_mines = 5, 5, 5  # Define the board size and number of mines
    env = MinesweeperEnv(width, height, n_mines)
    
    print("=== Welcome to Minesweeper ===")
    print("Input format: row column (starting from 0), for example `2 3`")
    print("Type `exit` to quit the game")

    while True:
        print("\nCurrent board:")
        print_board(env.state)  # Show the visible board to the player

        # User input
        move = input("Enter coordinates (row column): ").strip()
        if move.lower() == "exit":
            print("Game over!")
            break

        try:
            row, col = map(int, move.split())
            if row < 0 or col < 0 or row >= width or col >= height:
                print("Out of bounds, please try again!")
                continue
        except ValueError:
            print("Invalid input format, please enter two integers, for example `2 3`")
            continue

        index = row * width + col
        
        # Execute the dig action
        state, reward, done = env.step(index)

        # Show result
        print(f"Digging at ({row}, {col}), reward: {reward}")

        if done:
            if reward > 0:
                print("\n?? Congratulations, you won!")
            else:
                print("\n?? You hit a mine, game over!")
            # print("Final board:")
            # print_board(env.board)  # Show the full board
            # break

def print_board(state):
    """Format and print the board."""
    rows = []
    for row in range(5):  # Assuming the board is 5x5 based on the provided output
        row_values = []
        for col in range(5):
            coord = (row, col)
            # Find the corresponding 'value' for the coordinate
            cell = next(item['value'] for item in state if item['coord'] == coord)
            row_values.append(cell)
        row_values = [str(rv) for rv in row_values]
        rows.append(" ".join(row_values))
    
    # Print the formatted rows of the board
    for row in rows:
        print(row)
    
if __name__ == "__main__":
    play_minesweeper()
