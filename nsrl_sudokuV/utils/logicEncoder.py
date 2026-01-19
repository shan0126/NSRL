from pysdd.sdd import SddManager, Vtree
from pysdd import sdd
import pydot

import numpy as np

from domlogic.formula_sudoku2 import formula_sudoku2
from domlogic.formula_sudoku3 import formula_sudoku3
from domlogic.formula_sudoku4 import formula_sudoku4
from domlogic.formula_sudoku5 import formula_sudoku5
from domlogic.formula_sudoku4r import formula_sudoku4r
from domlogic.formula_sudoku6r import formula_sudoku6r
from domlogic.formula_sudoku9r import formula_sudoku9r
from domlogic.verify_sudoku import verify_sudoku
from simple_test import formula_sample

if __name__ == "__main__":
    formula_sample()
    verify_sudoku('SDDCircuits/sample/vtree.vtree', 'SDDCircuits/sample/sdd.sdd')
