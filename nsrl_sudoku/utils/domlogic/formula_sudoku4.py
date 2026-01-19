from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np

def formula_sudoku4():

    foldername = "SDDCircuits/sudoku4/b1/"
    
    preset_grid = [
        [4, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 3]
    ]
    
    var_count = 64  # 4x4 grid × 4 values
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    print("constructing SDD ... ")

    # Create a[i][j][k]: i=row, j=col, k=value (0-based)
    a = [[[None for k in range(4)] for j in range(4)] for i in range(4)]
    literal_id = 1
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a[i][j][k] = manager.literal(literal_id)
                literal_id += 1

    clauses = []

    # Each cell has at least one value
    for i in range(4):
        for j in range(4):
            clauses.append(a[i][j][0] | a[i][j][1] | a[i][j][2] | a[i][j][3])

    # Each cell has at most one value
    for i in range(4):
        for j in range(4):
            for k1 in range(4):
                for k2 in range(k1 + 1, 4):
                    clauses.append(~a[i][j][k1] | ~a[i][j][k2])

    # Each value appears once per row
    for i in range(4):
        for k in range(4):
            for j1 in range(4):
                for j2 in range(j1 + 1, 4):
                    clauses.append(~a[i][j1][k] | ~a[i][j2][k])

    # Each value appears once per column
    for j in range(4):
        for k in range(4):
            for i1 in range(4):
                for i2 in range(i1 + 1, 4):
                    clauses.append(~a[i1][j][k] | ~a[i2][j][k])
                    
    for i in range(4):
        for j in range(4):
            val = preset_grid[i][j]
            if val != 0:
                # k = val - 1 because value 1~4 → index 0~3
                clauses.append(a[i][j][val - 1])
    
    

    # Combine all clauses
    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause

    print("done")

    # Save SDD and vtree
    print("saving sdd and vtree as dot ... ")
    with open(foldername + "sudoku4.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "sudoku4.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sudoku4.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"sudoku4.vtree").encode())
    print("done")
    
    