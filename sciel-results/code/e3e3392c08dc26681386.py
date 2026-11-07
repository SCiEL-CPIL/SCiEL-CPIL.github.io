# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 diverse basis functions defined on [-2, 2].
    
    Basis:
      0: constant (1)
      1: linear (x)
      2: quadratic (x^2)
      3: cubic (x^3)
      4: sine (sin(pi*x/2))
      5: cosine (cos(pi*x/2))
      6: gaussian (exp(-x^2/2))
      7: tanh (tanh(x))
      8: exp_pos (exp(x/2))
      9: exp_neg (exp(-x/2))
    
    This basis covers polynomial, periodic, and non-linear exponential/saturating behaviors.
    """
    # Compute basis functions
    f0 = np.ones_like(x)
    f1 = x
    f2 = x**2
    f3 = x**3
    f4 = np.sin(np.pi * x / 2.0)
    f5 = np.cos(np.pi * x / 2.0)
    f6 = np.exp(-x**2 / 2.0)
    f7 = np.tanh(x)
    f8 = np.exp(x / 2.0)
    f9 = np.exp(-x / 2.0)
    
    # Stack basis functions
    basis = np.column_stack([
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9
    ])
    
    # Linear combination with params
    return basis @ params
# EVOLVE-BLOCK-END
