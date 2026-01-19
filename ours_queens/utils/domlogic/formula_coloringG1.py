from pysdd.sdd import SddManager, Vtree
from itertools import combinations

def formula_coloringG1():
    foldername = "SDDCircuits/coloringG1/"
    
    num_nodes = 8
    num_colors = 4
    var_count = num_nodes * num_colors
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    def var(node, color):
        """Return the literal that node uses this color"""
        return manager.literal(node * num_colors + color + 1)

    # 图结构：邻接表
    adjacency = {
        0: [1, 2, 7, 5],
        1: [0, 2, 3, 7, 6, 4],
        2: [0, 1, 3, 4, 5],
        3: [1, 2, 4, 6],
        4: [2, 3, 5, 6, 7],
        5: [2, 4, 6, 7, 0],
        6: [3, 4, 5, 7, 1],
        7: [0, 1, 4, 5, 6],
    }

    clauses = []

    # 每个节点至少有一个颜色
    for node in range(num_nodes):
        clause = var(node, 0)
        for c in range(1, num_colors):
            clause |= var(node, c)
        clauses.append(clause)

    # 每个节点至多一个颜色（任意两个颜色不能同时选）
    for node in range(num_nodes):
        for c1, c2 in combinations(range(num_colors), 2):
            clauses.append(~var(node, c1) | ~var(node, c2))

    # 邻接节点不能同色
    added = set()
    for node in range(num_nodes):
        for neighbor in adjacency[node]:
            if (neighbor, node) in added:  # skip symmetric pair
                continue
            for c in range(num_colors):
                clauses.append(~var(node, c) | ~var(neighbor, c))
            added.add((node, neighbor))

    # AND 所有约束成 α
    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause
    
    alpha &= var(0, 0)
    alpha &= var(1, 1)
        
    print("Model counting: {}".format(alpha.model_count()))

    # 保存 SDD 和 Vtree
    print("saving sdd and vtree as dot ...")
    with open(foldername + "coloringG1_sdd.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "coloringG1_vtree.dot", "w") as out:
        print(vtree.dot(), file=out)

    print("saving as sdd ...")
    alpha.save((foldername + "coloringG1.sdd").encode())
    print("saving as vtree ...")
    vtree.save((foldername + "coloringG1.vtree").encode())

    print("done")