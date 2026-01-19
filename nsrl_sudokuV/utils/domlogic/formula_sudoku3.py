from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np

def formula_sudoku3():
    foldername = "SDDCircuits/sudoku3/b1/"
    
    preset = [
        [2, 0, 0],
        [0, 0, 0],
        [0, 2, 0]
    ]
    
    var_count = 27  # 3x3 grid with 3 values (1-3)
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    print("constructing SDD ... ")

    # Step 1: Create literals a[i][j][k]
    a = [[[None for k in range(3)] for j in range(3)] for i in range(3)]
    literal_id = 1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a[i][j][k] = manager.literal(literal_id)
                literal_id += 1

    # Step 2: Build formula (conjunction of constraints)
    clauses = []

    # Constraint 1: Each cell has at least one number
    for i in range(3):
        for j in range(3):
            clause = a[i][j][0] | a[i][j][1] | a[i][j][2]
            clauses.append(clause)

    # Constraint 2: Each cell has at most one number
    for i in range(3):
        for j in range(3):
            for k1 in range(3):
                for k2 in range(k1 + 1, 3):
                    clause = ~a[i][j][k1] | ~a[i][j][k2]
                    clauses.append(clause)

    # Constraint 3: No repeated number in any row
    for i in range(3):  # row
        for k in range(3):  # value
            for j1 in range(3):
                for j2 in range(j1 + 1, 3):
                    clause = ~a[i][j1][k] | ~a[i][j2][k]
                    clauses.append(clause)

    # Constraint 4: No repeated number in any column
    for j in range(3):  # column
        for k in range(3):  # value
            for i1 in range(3):
                for i2 in range(i1 + 1, 3):
                    clause = ~a[i1][j][k] | ~a[i2][j][k]
                    clauses.append(clause)
                    
    for i in range(3):
        for j in range(3):
            val = preset[i][j]
            if val != 0:
                # preset value: (i, j) is val → a[i][j][val-1] must be true
                clauses.append(a[i][j][val - 1])

    # Combine all clauses
    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause

    print("done")

    # Save dot files
    print("saving sdd and vtree as dot ... ")
    with open(foldername + "sdd_sudoku3.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "vtree_sudoku3.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sudoku3.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"sudoku3.vtree").encode())
    print("done")
    
    print("write sdd as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"sdd_sudoku3.dot")
    graph = graph[0]
    graph.write_png(foldername+'sdd_sudoku3.png')
    
    print("write vtree as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"vtree_sudoku3.dot")
    graph = graph[0]
    graph.write_png(foldername+'vtree_sudoku3.png')
    print("done")


def formula_sudoku3_hand():

    foldername = "SDDCircuits/sudoku3/"
    
    var_count = 27
    var_order = [i for i in range(1, var_count+1)]
    vtree_type = "balanced"
    
    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)
    
    print("constructing SDD ... ")
    a_1_1_1, a_1_1_2, a_1_1_3, \
    a_2_1_1, a_2_1_2, a_2_1_3, \
    a_3_1_1, a_3_1_2, a_3_1_3, \
    a_1_2_1, a_1_2_2, a_1_2_3, \
    a_2_2_1, a_2_2_2, a_2_2_3, \
    a_3_2_1, a_3_2_2, a_3_2_3,\
    a_1_3_1, a_1_3_2, a_1_3_3, \
    a_2_3_1, a_2_3_2, a_2_3_3, \
    a_3_3_1, a_3_3_2, a_3_3_3 = [manager.literal(i) for i in range(1, 28)]
    
    alpha = (a_1_1_1 | a_1_1_2 | a_1_1_3) & \
            (a_2_1_1 | a_2_1_2 | a_2_1_3) & \
            (a_3_1_1 | a_3_1_2 | a_3_1_3) & \
            (a_1_2_1 | a_1_2_2 | a_1_2_3) & \
            (a_2_2_1 | a_2_2_2 | a_2_2_3) & \
            (a_3_2_1 | a_3_2_2 | a_3_2_3) & \
            (a_1_3_1 | a_1_3_2 | a_1_3_3) & \
            (a_2_3_1 | a_2_3_2 | a_2_3_3) & \
            (a_3_3_1 | a_3_3_2 | a_3_3_3) & \
            (~a_1_1_1 | ~a_1_1_2) & (~a_1_1_1 | ~a_1_1_3) & (~a_1_1_2 | ~a_1_1_3) & \
            (~a_1_2_1 | ~a_1_2_2) & (~a_1_2_1 | ~a_1_2_3) & (~a_1_2_2 | ~a_1_2_3) & \
            (~a_1_3_1 | ~a_1_3_2) & (~a_1_3_1 | ~a_1_3_3) & (~a_1_3_2 | ~a_1_3_3) & \
            (~a_2_1_1 | ~a_2_1_2) & (~a_2_1_1 | ~a_2_1_3) & (~a_2_1_2 | ~a_2_1_3) & \
            (~a_2_2_1 | ~a_2_2_2) & (~a_2_2_1 | ~a_2_2_3) & (~a_2_2_2 | ~a_2_2_3) & \
            (~a_2_3_1 | ~a_2_3_2) & (~a_2_3_1 | ~a_2_3_3) & (~a_2_3_2 | ~a_2_3_3) & \
            (~a_3_1_1 | ~a_3_1_2) & (~a_3_1_1 | ~a_3_1_3) & (~a_3_1_2 | ~a_3_1_3) & \
            (~a_3_2_1 | ~a_3_2_2) & (~a_3_2_1 | ~a_3_2_3) & (~a_3_2_2 | ~a_3_2_3) & \
            (~a_3_3_1 | ~a_3_3_2) & (~a_3_3_1 | ~a_3_3_3) & (~a_3_3_2 | ~a_3_3_3) & \
            (~a_1_1_1 | ~a_1_2_1) & (~a_1_1_1 | ~a_1_3_1) & (~a_1_2_1 | ~a_1_3_1) & \
            (~a_2_1_1 | ~a_2_2_1) & (~a_2_1_1 | ~a_2_3_1) & (~a_2_2_1 | ~a_2_3_1) & \
            (~a_3_1_1 | ~a_3_2_1) & (~a_3_1_1 | ~a_3_3_1) & (~a_3_2_1 | ~a_3_3_1) & \
            (~a_1_1_1 | ~a_2_1_1) & (~a_1_1_1 | ~a_3_1_1) & (~a_2_1_1 | ~a_3_1_1) & \
            (~a_1_2_1 | ~a_2_2_1) & (~a_1_2_1 | ~a_3_2_1) & (~a_2_2_1 | ~a_3_2_1) & \
            (~a_1_3_1 | ~a_2_3_1) & (~a_1_3_1 | ~a_3_3_1) & (~a_2_3_1 | ~a_3_3_1) & \
            (~a_1_1_2 | ~a_1_2_2) & (~a_1_1_2 | ~a_1_3_2) & (~a_1_2_2 | ~a_1_3_2) & \
            (~a_2_1_2 | ~a_2_2_2) & (~a_2_1_2 | ~a_2_3_2) & (~a_2_2_2 | ~a_2_3_2) & \
            (~a_3_1_2 | ~a_3_2_2) & (~a_3_1_2 | ~a_3_3_2) & (~a_3_2_2 | ~a_3_3_2) & \
            (~a_1_1_2 | ~a_2_1_2) & (~a_1_1_2 | ~a_3_1_2) & (~a_2_1_2 | ~a_3_1_2) & \
            (~a_1_2_2 | ~a_2_2_2) & (~a_1_2_2 | ~a_3_2_2) & (~a_2_2_2 | ~a_3_2_2) & \
            (~a_1_3_2 | ~a_2_3_2) & (~a_1_3_2 | ~a_3_3_2) & (~a_2_3_2 | ~a_3_3_2) & \
            (~a_1_1_3 | ~a_1_2_3) & (~a_1_1_3 | ~a_1_3_3) & (~a_1_2_3 | ~a_1_3_3) & \
            (~a_2_1_3 | ~a_2_2_3) & (~a_2_1_3 | ~a_2_3_3) & (~a_2_2_3 | ~a_2_3_3) & \
            (~a_3_1_3 | ~a_3_2_3) & (~a_3_1_3 | ~a_3_3_3) & (~a_3_2_3 | ~a_3_3_3) & \
            (~a_1_1_3 | ~a_2_1_3) & (~a_1_1_3 | ~a_3_1_3) & (~a_2_1_3 | ~a_3_1_3) & \
            (~a_1_2_3 | ~a_2_2_3) & (~a_1_2_3 | ~a_3_2_3) & (~a_2_2_3 | ~a_3_2_3) & \
            (~a_1_3_3 | ~a_2_3_3) & (~a_1_3_3 | ~a_3_3_3) & (~a_2_3_3 | ~a_3_3_3)
            
    
    
    print("done")
    
    print("saving sdd and vtree as dot ... ")
    with open(foldername+"sdd_sudoku3.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername+"vtree_sudoku3.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sudoku3.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"sudoku3.vtree").encode())
    print("done")
    
    print("write sdd as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"sdd_sudoku3.dot")
    graph = graph[0]
    graph.write_png(foldername+'sdd_sudoku3.png')
    
    print("write vtree as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"vtree_sudoku3.dot")
    graph = graph[0]
    graph.write_png(foldername+'vtree_sudoku3.png')
    print("done")
    
    