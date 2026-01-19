from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np

def formula_sudoku6r():
    foldername = "SDDCircuits/sudoku6r/b1/"
    
    # Add initial clues: (row, col, value)
    preset1 = [
        (0, 0, 0),  # (1,1) = 1
        (1, 1, 1),  # (2,2) = 2
        (2, 2, 2),  # (3,3) = 3
        (3, 3, 3),  # (4,4) = 4
        (4, 4, 4),  # (5,5) = 5
        (5, 5, 5)   # (6,6) = 6
    ]
    
    var_count = 36 * 6  # 6x6 grid, each cell has 6 possible values
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    print("constructing SDD ... ")

    # Create literals: a[i][j][k] -> i=row, j=col, k=value
    a = [[[None for k in range(6)] for j in range(6)] for i in range(6)]
    literal_id = 1
    for i in range(6):
        for j in range(6):
            for k in range(6):
                a[i][j][k] = manager.literal(literal_id)
                literal_id += 1

    clauses = []

    # Constraint 1: each cell has at least one number
    for i in range(6):
        for j in range(6):
            clauses.append(a[i][j][0] | a[i][j][1] | a[i][j][2] | a[i][j][3] | a[i][j][4] | a[i][j][5])

    # Constraint 2: each cell has at most one number
    for i in range(6):
        for j in range(6):
            for k1 in range(6):
                for k2 in range(k1 + 1, 6):
                    clauses.append(~a[i][j][k1] | ~a[i][j][k2])

    # Constraint 3: each number appears at most once per row
    for i in range(6):
        for k in range(6):
            for j1 in range(6):
                for j2 in range(j1 + 1, 6):
                    clauses.append(~a[i][j1][k] | ~a[i][j2][k])

    # Constraint 4: each number appears at most once per column
    for j in range(6):
        for k in range(6):
            for i1 in range(6):
                for i2 in range(i1 + 1, 6):
                    clauses.append(~a[i1][j][k] | ~a[i2][j][k])

    # Constraint 5: each number appears at most once in each 2x2 block
    for block_i in [0, 2, 4]:  # Blocks: starting from 0, 2, and 4 in both row and column
        for block_j in [0, 2, 4]:
            for k in range(6):
                block_cells = [(i, j) for i in range(block_i, block_i + 2) for j in range(block_j, block_j + 2)]
                for idx1 in range(len(block_cells)):
                    for idx2 in range(idx1 + 1, len(block_cells)):
                        i1, j1 = block_cells[idx1]
                        i2, j2 = block_cells[idx2]
                        clauses.append(~a[i1][j1][k] | ~a[i2][j2][k])

    # Apply initial clues
    for i, j, k in preset1:
        clauses.append(a[i][j][k])  # a[i][j][k] must be True

    # Combine all clauses
    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause

    print("done")

    # Save SDD and Vtree to .dot files
    print("saving sdd and vtree as dot ... ")
    with open(foldername + "sdd_sudoku6.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "vtree_sudoku6.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sudoku6.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"sudoku6.vtree").encode())
    print("done")
    