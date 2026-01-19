import sys
sys.path.insert(0, '/gpfs/home4/shuaih/file2025_ns/005_supervised/004_supervised/nsrl/pypsdd/')

from pypsdd import Vtree, SddManager, PSddManager, SddNode, Inst, io
from pypsdd import UniformSmoothing, Prior



def verify_sudoku(vtree_2s3z, sdd_2s3z):
    # *
    vtree = Vtree.read(vtree_2s3z)
    manager = SddManager(vtree)
    # *
    alpha = io.sdd_read(sdd_2s3z, manager)
    

    pmanager = PSddManager(vtree)
    beta = pmanager.copy_and_normalize_sdd(alpha, vtree)
    print("============= Verify the model by counting =============")
    print("model_counting:{}".format(beta.model_count()))


    
    


