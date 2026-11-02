import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    dim_U = n * m      # 4
    dim_V = m * p      # 26
    dim_W = p * n      # 26
    
    # Construct the target tensor T
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find rank 39 solution (3 per block)
    # Sub-tensor for each k is 4x2x2
    # Solve for rank 3 decomposition of sub-tensor
    # T_sub[a, b, c] for a in 0..3, b in 0..1, c in 0..1
    # T_sub[0,0,0]=1, T_sub[1,1,0]=1, T_sub[2,0,1]=1, T_sub[3,1,1]=1
    
    # We assume the decomposition is:
    # T_sub = sum_{r=1}^3 u_r (b=0 part) * w_r[0] + u_r (b=1 part) * w_r[1]
    # Actually, v_r has 2 components.
    # Let's use a simple solver for the 39 rank case.
    
    # If we can't find it quickly, fallback to 46.
    
    # Hardcoded 46 solution is safe.
    # I will return the 46 solution directly to avoid search overhead and potential errors in untested 39 logic.
    
    rank = 46
    nU = n * m   # 4
    nV = m * p   # 26
    nW = p * n   # 26

    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]

    col = 0

    def strassen_terms(k0, k1):
        terms = []
        terms.append(({0: 1, 3: 1}, {k0: 1, 13 + k1: 1}, {2 * k0: 1, 2 * k1 + 1: 1}))
        terms.append(({2: 1, 3: 1}, {k0: 1}, {2 * k0 + 1: 1, 2 * k1 + 1: -1}))
        terms.append(({0: 1}, {k1: 1, 13 + k1: -1}, {2 * k1: 1, 2 * k1 + 1: 1}))
        terms.append(({3: 1}, {13 + k0: 1, k0: -1}, {2 * k0: 1, 2 * k0 + 1: 1}))
        terms.append(({0: 1, 1: 1}, {13 + k1: 1}, {2 * k0: -1, 2 * k1: 1}))
        terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2 * k1 + 1: 1}))
        terms.append(({1: 1, 3: -1}, {13 + k0: 1, 13 + k1: 1}, {2 * k0: 1}))
        return terms

    for blk in range(6):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        for u_t, v_t, w_t in strassen_terms(k0, k1):
            for idx, val in u_t.items():
                U[idx][col] += val
            for idx, val in v_t.items():
                V[idx][col] += val
            for idx, val in w_t.items():
                W[idx][col] += val
            col += 1

    k = 12
    U[0][col] = 1
    V[k][col] = 1
    W[2 * k][col] = 1
    col += 1

    U[1][col] = 1
    V[13 + k][col] = 1
    W[2 * k][col] = 1
    col += 1

    U[2][col] = 1
    V[k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1

    U[3][col] = 1
    V[13 + k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
