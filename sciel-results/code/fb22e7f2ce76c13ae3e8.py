import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    rank = 1
    U = [[1.0] * rank for _ in range(9)]
    V = [[1.0] * rank for _ in range(9)]
    W = [[1.0] * rank for _ in range(9)]
    return {"U": U, "V": V, "W": W}
