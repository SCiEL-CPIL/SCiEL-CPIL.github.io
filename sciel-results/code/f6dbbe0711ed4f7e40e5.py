"""Symbolic regression with 1 input - Fixed-slope Tanh Basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model with a linear term and 4 Tanh components with fixed slopes.
    
    y = p0 + p1*x + p2*tanh(1*x + p3) + p4*tanh(2*x + p5) + p6*tanh(3*x + p7) + p8*tanh(4*x + p9)
    
    Parameters:
      params[0]: bias
      params[1]: linear coefficient
      params[2]: weight for tanh(x + p3)
      params[3]: bias for tanh(x + p3)
      params[4]: weight for tanh(2x + p5)
      params[5]: bias for tanh(2x + p5)
      params[6]: weight for tanh(3x + p7)
      params[7]: bias for tanh(3x + p7)
      params[8]: weight for tanh(4x + p9)
      params[9]: bias for tanh(4x + p9)
    """
    # Linear part
    y = params[0] + params[1] * x
    
    # Tanh parts
    # Slopes are fixed to 1, 2, 3, 4
    y += params[2] * np.tanh(1.0 * x + params[3])
    y += params[4] * np.tanh(2.0 * x + params[5])
    y += params[6] * np.tanh(3.0 * x + params[7])
    y += params[8] * np.tanh(4.0 * x + params[9])
    
    return y
# EVOLVE-BLOCK-END
