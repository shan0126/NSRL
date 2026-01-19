from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np

def formula_sample():

    foldername = "SDDCircuits/sample/"
    
    var_count = 3
    var_order = [1, 2, 3]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    print("constructing SDD ...")
    
    a_1, a_2, a_3 = [manager.literal(i) for i in range(1, 4)]

    # ?a_1 กล a_3
    imp1 = ~a_3 | a_1
    # ?a_2 กล a_3
    imp2 = ~a_2 | a_1
    xor = (a_3 & ~a_2) | (~a_3 & a_2)

    alpha = imp1 & imp2 & xor
    
    print("saving sdd and vtree as dot ... ")
    with open(foldername+"sdd.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername+"vtree.dot", "w") as out:
        print(vtree.dot(), file=out)
    print("done")
    
    print("saving as sdd ... ")
    alpha.save((foldername+"sdd.sdd").encode())
    #sdd.sdd_save_as_dot(filename +".sdd.dot",alpha)
    print("saving as vtree ... ")
    vtree.save((foldername+"vtree.vtree").encode())
    print("done")
    
    print("write sdd as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"sdd.dot")
    graph = graph[0]
    graph.write_png(foldername+'sdd.png')
    
    print("write vtree as png ... ")
    graph = pydot.graph_from_dot_file(foldername+"vtree.dot")
    graph = graph[0]
    graph.write_png(foldername+'vtree.png')
    print("done")
    
    