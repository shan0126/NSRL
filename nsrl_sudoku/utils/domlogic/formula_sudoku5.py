from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np

def formula_sudoku5():
    
    foldername = "SDDCircuits/sudoku5/b1/"
    
    preset_grid = [
        [0, 0, 5, 0, 1],
        [0, 0, 0, 0, 3],
        [0, 1, 2, 0, 0],
        [0, 0, 0, 0, 4],
        [0, 0, 0, 3, 0]
    ]

    var_count = 125  # 5x5 grid ¡Á 5 values
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    print("constructing SDD ... ")

    # Create literals: a[i][j][k] = (i,j) filled with value (k+1)
    a = [[[None for k in range(5)] for j in range(5)] for i in range(5)]
    literal_id = 1
    for i in range(5):
        for j in range(5):
            for k in range(5):
                a[i][j][k] = manager.literal(literal_id)
                literal_id += 1

    clauses = []

    # 1. Each cell has at least one number
    for i in range(5):
        for j in range(5):
            clause = a[i][j][0]
            for k in range(1, 5):
                clause |= a[i][j][k]
            clauses.append(clause)

    # 2. Each cell has at most one number (mutual exclusion)
    for i in range(5):
        for j in range(5):
            for k1 in range(5):
                for k2 in range(k1 + 1, 5):
                    clauses.append(~a[i][j][k1] | ~a[i][j][k2])

    # 3. Each value appears at most once per row
    for i in range(5):
        for k in range(5):
            for j1 in range(5):
                for j2 in range(j1 + 1, 5):
                    clauses.append(~a[i][j1][k] | ~a[i][j2][k])

    # 4. Each value appears at most once per column
    for j in range(5):
        for k in range(5):
            for i1 in range(5):
                for i2 in range(i1 + 1, 5):
                    clauses.append(~a[i1][j][k] | ~a[i2][j][k])
                    
    for i in range(5):
        for j in range(5):
            val = preset_grid[i][j]
            if val != 0:
                clauses.append(a[i][j][val - 1])  # Adjust for 0-based index
    

    # Combine clauses
    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause

    print("done")

    # Save dot files
    print("saving sdd and vtree as dot ... ")
    with open(foldername + "sdd_sudoku5.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "vtree_sudoku5.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sudoku5.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"sudoku5.vtree").encode())
    print("done")
    
    