"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters to recover a hidden function.
    Basis: constant, linear (x,y,z), log(x,y,z), exp(x-1), exp(y-1), exp(z-1)
    """
    p = params
    
    # Compute basis terms
    t1 = np.ones_like(x)          # constant
    t2 = x                        # linear x
    t3 = y                        # linear y
    t4 = z                        # linear z
    t5 = np.log(x)                # log x
    t6 = np.log(y)                # log y
    t7 = np.log(z)                # log z
    t8 = np.exp(x - 1.0)          # exp(x-1)
    t9 = np.exp(y - 1.0)          # exp(y-1)
    t10 = np.exp(z - 1.0)         # exp(z-1)
    
    return (p[0] * t1 + 
            p[1] * t2 + 
            p[2] * t3 + 
            p[3] * t4 + 
            p[4] * t5 + 
            p[5] * t6 + 
            p[6] * t7 + 
            p[7] * t8 + 
            p[8] * t9 + 
            p[9] * t10)
# EVOLVE-BLOCK-END
