from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np

def formula_sudoku2():

    foldername = "SDDCircuits/sudoku2/b1/"
    
    var_count = 8
    var_order = [i for i in range(1, var_count+1)]
    vtree_type = "balanced"
    
    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)
    
    print("constructing SDD ... ")
    a_1_1_1, a_1_1_2, a_2_1_1, a_2_1_2, a_1_2_1, a_1_2_2, a_2_2_1, a_2_2_2 = [manager.literal(i) for i in range(1, 9)]
    
    alpha = (a_1_1_1 | a_1_1_2) & (a_2_1_1 | a_2_1_2) & (a_1_2_1 | a_1_2_2) & (a_2_2_1 | a_2_2_2) & (~a_1_1_2 | ~a_1_1_1) & (~a_2_1_2 | ~a_2_1_1) & (~a_1_2_2 | ~a_1_2_1) & (~a_2_2_2 | ~a_2_2_1) & (~a_1_1_1 | ~a_2_1_1) & (~a_2_1_1 | ~a_2_2_1) & (~a_1_1_1 | ~a_1_2_1) & (~a_1_2_1 | ~a_2_2_1) & (~a_1_1_2 | ~a_2_1_2) & (~a_2_1_2 | ~a_2_2_2) & (~a_1_1_2 | ~a_1_2_2) & (~a_1_2_2 | ~a_2_2_2)
    
    print("done")
    
    print("saving sdd and vtree as dot ... ")
    with open(foldername+"sdd_sudoku2.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername+"vtree_sudoku2.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sudoku2.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"sudoku2.vtree").encode())
    print("done")
    
    print("write sdd as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"sdd_sudoku2.dot")
    graph = graph[0]
    graph.write_png(foldername+'sdd_sudoku2.png')
    
    print("write vtree as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"vtree_sudoku2.dot")
    graph = graph[0]
    graph.write_png(foldername+'vtree_sudoku2.png')
    print("done")
    
    