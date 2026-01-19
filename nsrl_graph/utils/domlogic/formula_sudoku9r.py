from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np


def formula_sudoku9r():
    foldername = "SDDCircuits/sudoku9r/"
    
    preset_grid = [
        [0, 2, 0, 6, 0, 8, 0, 0, 0],
        [5, 8, 0, 0, 0, 9, 7, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0],
        [3, 7, 0, 0, 0, 0, 5, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 8, 0, 0, 0, 0, 1, 3],
        [0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 9, 8, 0, 0, 0, 3, 6],
        [0, 0, 0, 3, 0, 6, 0, 9, 0]
    ]

    
    var_count = 729  # 9x9 grid with 9 values per cell
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    print("constructing SDD ... ")

    # Create literals: a[i][j][k]
    a = [[[None for k in range(9)] for j in range(9)] for i in range(9)]
    literal_id = 1
    for i in range(9):
        for j in range(9):
            for k in range(9):
                a[i][j][k] = manager.literal(literal_id)
                literal_id += 1

    clauses = []

    # Constraint 1: each cell has at least one value
    for i in range(9):
        for j in range(9):
            clause = a[i][j][0]
            for k in range(1, 9):
                clause |= a[i][j][k]
            clauses.append(clause)

    # Constraint 2: each cell has at most one value
    for i in range(9):
        for j in range(9):
            for k1 in range(9):
                for k2 in range(k1 + 1, 9):
                    clauses.append(~a[i][j][k1] | ~a[i][j][k2])

    # Constraint 3: each value appears once per row
    for i in range(9):
        for k in range(9):
            for j1 in range(9):
                for j2 in range(j1 + 1, 9):
                    clauses.append(~a[i][j1][k] | ~a[i][j2][k])

    # Constraint 4: each value appears once per column
    for j in range(9):
        for k in range(9):
            for i1 in range(9):
                for i2 in range(i1 + 1, 9):
                    clauses.append(~a[i1][j][k] | ~a[i2][j][k])

    # Constraint 5: each value appears once per 3x3 box
    for box_row in range(3):
        for box_col in range(3):
            for k in range(9):
                cells = [(i, j) for i in range(box_row*3, box_row*3 + 3)
                                for j in range(box_col*3, box_col*3 + 3)]
                for idx1 in range(len(cells)):
                    for idx2 in range(idx1 + 1, len(cells)):
                        i1, j1 = cells[idx1]
                        i2, j2 = cells[idx2]
                        clauses.append(~a[i1][j1][k] | ~a[i2][j2][k])

    # Add fixed values to clauses
    for i in range(9):
        for j in range(9):
            val = preset_grid[i][j]
            if val != 0:
                clauses.append(a[i][j][val - 1])  # Because k index is 0-based
    
    # Combine all constraints into a single formula
    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause

    print("done")

    # Save to .dot for visualization
    print("saving sdd and vtree as dot ... ")
    with open(foldername + "sdd_sudoku9.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "vtree_sudoku9.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sudoku9.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"sudoku9.vtree").encode())
    print("done")
    
    