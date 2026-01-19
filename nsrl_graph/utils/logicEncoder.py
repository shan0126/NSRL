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
from domlogic.formula_queens4 import formula_nqueens4, formula_nqueens4_partial
from domlogic.formula_queens6 import formula_nqueens6
from domlogic.formula_queens8 import formula_nqueens8
from domlogic.formula_queens10 import formula_nqueens10
from domlogic.formula_coloringG1 import formula_coloringG1
from domlogic.formula_coloringG2 import formula_coloringG2
from domlogic.formula_coloringG3 import formula_coloringG3
from domlogic.formula_coloringG4 import formula_coloringG4
from simple_test import formula_sample

if __name__ == "__main__":
    formula_nqueens10()
    verify_sudoku('SDDCircuits/nqueens10/nqueens10.vtree', 'SDDCircuits/nqueens10/nqueens10.sdd')
