import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Uses a block decomposition into 6 Strassen 2x2x2 multiplications, yielding rank 42.
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    rank = 42
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 coefficients
    # Terms:
    # 0: M1 = (a00+a11)(b00+b11) -> C00+=1, C11+=1
    # 1: M2 = (a10+a11)b00     -> C10+=1, C11-=1
    # 2: M3 = a00(b01-b11)     -> C01+=1, C11+=1
    # 3: M4 = a11(b10-b00)     -> C00+=1, C10+=1
    # 4: M5 = (a00+a01)b11     -> C00-=1, C01+=1
    # 5: M6 = (a10-a00)(b00+b01)-> C11+=1
    # 6: M7 = (a01-a11)(b10+b11)-> C00+=1
    
    # U indices: i * m + (2*j_b + j_loc)
    # V indices: (2*j_b + j_loc) * p + (2*k + k_loc)
    # W indices: (2*k + k_loc) * n + i
    
    for k in range(3):
        for j_b in range(2):
            for r in range(7):
                c = (k * 2 + j_b) * 7 + r
                
                if r == 0:
                    # M1
                    U[0 * m + 2 * j_b + 0, c] = 1
                    U[1 * m + 2 * j_b + 1, c] = 1
                    V[(2 * j_b + 0) * p + 2 * k + 0, c] = 1
                    V[(2 * j_b + 1) * p + 2 * k + 1, c] = 1
                    W[(2 * k + 0) * n + 0, c] = 1
                    W[(2 * k + 1) * n + 1, c] = 1
                elif r == 1:
                    # M2
                    U[1 * m + 2 * j_b + 0, c] = 1
                    U[1 * m + 2 * j_b + 1, c] = 1
                    V[(2 * j_b + 0) * p + 2 * k + 0, c] = 1
                    W[(2 * k + 0) * n + 1, c] = 1
                    W[(2 * k + 1) * n + 1, c] = -1
                elif r == 2:
                    # M3
                    U[0 * m + 2 * j_b + 0, c] = 1
                    V[(2 * j_b + 0) * p + 2 * k + 1, c] = 1
                    V[(2 * j_b + 1) * p + 2 * k + 1, c] = -1
                    W[(2 * k + 1) * n + 0, c] = 1
                    W[(2 * k + 1) * n + 1, c] = 1
                elif r == 3:
                    # M4
                    U[1 * m + 2 * j_b + 1, c] = 1
                    V[(2 * j_b + 1) * p + 2 * k + 0, c] = 1
                    V[(2 * j_b + 0) * p + 2 * k + 0, c] = -1
                    W[(2 * k + 0) * n + 0, c] = 1
                    W[(2 * k + 0) * n + 1, c] = 1
                elif r == 4:
                    # M5
                    U[0 * m + 2 * j_b + 0, c] = 1
                    U[0 * m + 2 * j_b + 1, c] = 1
                    V[(2 * j_b + 1) * p + 2 * k + 1, c] = 1
                    W[(2 * k + 0) * n + 0, c] = -1
                    W[(2 * k + 1) * n + 0, c] = 1
                elif r == 5:
                    # M6
                    U[1 * m + 2 * j_b + 0, c] = 1
                    U[0 * m + 2 * j_b + 0, c] = -1
                    V[(2 * j_b + 0) * p + 2 * k + 0, c] = 1
                    V[(2 * j_b + 0) * p + 2 * k + 1, c] = 1
                    W[(2 * k + 1) * n + 1, c] = 1
                elif r == 6:
                    # M7
                    U[0 * m + 2 * j_b + 1, c] = 1
                    U[1 * m + 2 * j_b + 1, c] = -1
                    V[(2 * j_b + 1) * p + 2 * k + 0, c] = 1
                    V[(2 * j_b + 1) * p + 2 * k + 1, c] = 1
                    W[(2 * k + 0) * n + 0, c] = 1

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
