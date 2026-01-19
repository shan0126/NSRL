# Neuro-symbolic Action Masking

This is the code for "Neuro-symbolic Action Masking for Deep Reinforcement Learning," which is accepted in AAMAS 2026.

## Requirements
* python 3.7.9
* OpenAI Gym 0.15.7
* PyTorch 1.12.0
* pypsdd


## Run the code

As shown in the paper, NSRL is tested on four environments.

For sudoku environment:

    cd nsrl_sudoku
    sh sudoku2.sh
    sh sudoku3.sh
    sh sudoku4.sh
    sh sudoku5.sh

For graph coloring environment:

    cd nsrl_graph
    sh coloringG.sh
    sh coloringG2.sh
    sh coloringG3.sh
    sh coloringG4.sh
    
For n queens environment:

    cd nsrl_queens
    sh queen4.sh
    sh queen6.sh
    sh queen8.sh
    sh queen10.sh
    
For visual sudoku environment:

    cd nsrl_sudokuV
    sh test_1.sh
    sh test_2.sh
    sh test_3.sh
    sh test_4.sh

    
## Reference
The RL training framework is adapted from https://github.com/Lizhi-sjtu/DRL-code-pytorch

Please cite this work if you use our code:


