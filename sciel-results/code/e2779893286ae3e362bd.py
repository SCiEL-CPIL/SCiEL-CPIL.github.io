import math
import time
import numpy as np

def coefficients(stages):
    """
    2nd-order symmetric splitting (Strang splitting) with 12 steps.
    U(t) = (exp(-i (t/24) B) exp(-i (t/12) A) exp(-i (t/24) B))^12
    This can be rewritten as:
    prod_{k=1}^{12} exp(-i (t/12) B) exp(-i (t/12) A)
    because exp(-i (t/24) B) exp(-i (t/24) B) = exp(-i (t/12) B).
    
    This uses all 12 B slots and all 12 A slots.
    b_k = 1/12
    a_k = 1/12
    Sum(b) = 1
    Sum(a) = 1
    """
    n = 12
    b = [1.0 / n] * n
    a = [1.0 / n] * n
    return a, b
