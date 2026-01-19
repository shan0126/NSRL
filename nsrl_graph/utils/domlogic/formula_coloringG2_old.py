from pysdd.sdd import SddManager, Vtree
from itertools import combinations

def formula_coloringG2():
    foldername = "SDDCircuits/coloringG2/"
    
    num_nodes = 6
    num_colors = 3   # 例如用 3 种颜色
    var_count = num_nodes * num_colors
    var_order = [i for i in range(1, var_count + 1)]
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    def var(node, color):
        """Return the literal that node uses this color"""
        return manager.literal(node * num_colors + color + 1)

    # 图结构：6 节点环
    adjacency = {
        0: [1, 5],
        1: [0, 2, 4, 5],
        2: [1, 3],
        3: [2, 4, 1],
        4: [3, 5],
        5: [4, 0, 1],
    }

    clauses = []

    # 每个节点至少有一个颜色
    for node in range(num_nodes):
        clause = var(node, 0)
        for c in range(1, num_colors):
            clause |= var(node, c)
        clauses.append(clause)

    # 每个节点至多一个颜色
    for node in range(num_nodes):
        for c1, c2 in combinations(range(num_colors), 2):
            clauses.append(~var(node, c1) | ~var(node, c2))

    # 邻接节点不能同色
    added = set()
    for node in range(num_nodes):
        for neighbor in adjacency[node]:
            if (neighbor, node) in added:  # 跳过对称边
                continue
            for c in range(num_colors):
                clauses.append(~var(node, c) | ~var(neighbor, c))
            added.add((node, neighbor))

    # AND 所有约束成 α
    alpha = clauses[0]
    for clause in clauses[1:]:
        alpha &= clause

    # 这里可以加一些具体约束（例如固定某些节点颜色），也可以不加
    # alpha &= var(0, 0)  # 举例：固定 0 号节点用颜色 0

    print("Model counting: {}".format(alpha.model_count()))

    # 保存 SDD 和 Vtree
    print("saving sdd and vtree as dot ...")
    with open(foldername + "coloringG2_sdd.dot", "w") as out:
        print(alpha.dot(), file=out)
    with open(foldername + "coloringG2_vtree.dot", "w") as out:
        print(vtree.dot(), file=out)

    print("saving as sdd ...")
    alpha.save((foldername + "coloringG2.sdd").encode())
    print("saving as vtree ...")
    vtree.save((foldername + "coloringG2.vtree").encode())

    print("done")