"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining polynomial and trigonometric terms.
    
    The polynomial part (degree up to 3) captures smooth non-linearities and
    interactions, while the trigonometric part captures periodic behavior.
    This hybrid should perform well across a broader class of target functions
    than either pure polynomial or pure Fourier.
    
    params: array of length 10, fitted by BFGS.
    
    Basis terms:
      p0: 1 (constant)
      p1: x
      p2: y
      p3: x^2
      p4: y^2
      p5: x*y
      p6: sin(pi*x)
      p7: sin(pi*y)
      p8: cos(pi*x) * cos(pi*y)
      p9: sin(pi*(x+y))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Polynomial terms
    t0 = params[0]
    t1 = params[1] * x
    t2 = params[2] * y
    t3 = params[3] * x * x
    t4 = params[4] * y * y
    t5 = params[5] * x * y
    
    # Trigonometric terms
    pi = np.pi
    t6 = params[6] * np.sin(pi * x)
    t7 = params[7] * np.sin(pi * y)
    t8 = params[8] * np.cos(pi * x) * np.cos(pi * y)
    t9 = params[9] * np.sin(pi * (x + y))
    
    return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
# EVOLVE-BLOCK-END
