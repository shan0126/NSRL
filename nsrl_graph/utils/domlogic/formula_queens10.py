from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot
from itertools import combinations

def formula_nqueens10():
    foldername = "SDDCircuits/nqueens10/"
    
    n = 10
    var_count = n * n  
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    print("constructing SDD ...")

    def var(i, j):
        return manager.literal(i * n + j + 1)  

    clauses = []

    for i in range(n):
        row_vars = [var(i, j) for j in range(n)]
        clause = row_vars[0]
        for v in row_vars[1:]:
            clause |= v
        clauses.append(clause)
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                clauses.append(~var(i, j1) | ~var(i, j2))

    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                clauses.append(~var(i1, j) | ~var(i2, j))

    for d in range(-n + 1, n):
        diag = [(i, i - d) for i in range(n) if 0 <= i - d < n]
        for (i1, j1), (i2, j2) in combinations(diag, 2):
            clauses.append(~var(i1, j1) | ~var(i2, j2))

    for d in range(2 * n - 1):
        diag = [(i, d - i) for i in range(n) if 0 <= d - i < n]
        for (i1, j1), (i2, j2) in combinations(diag, 2):
            clauses.append(~var(i1, j1) | ~var(i2, j2))

    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause
    
    alpha &= var(0, 0)
    alpha &= var(5, 3)


    print("done")

    print("saving sdd and vtree as dot ...")
    with open(foldername + "nqueens10_sdd.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "nqueens10_vtree.dot", "w") as out:
       print(vtree.dot(), file=out)

    print("saving as sdd ...")
    alpha.save((foldername + "nqueens10.sdd").encode())
    print("saving as vtree ...")
    vtree.save((foldername + "nqueens10.vtree").encode())

    print("done")