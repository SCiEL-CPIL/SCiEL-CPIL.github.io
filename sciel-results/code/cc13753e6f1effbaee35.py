"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using block Strassen
    # Split middle dim m=5 into j-groups: {0,1}, {2}, {3,4}
    # Split output dim p=6 into k-groups: {0,1}, {2,3}, {4,5}
    # For each (j-group, k-group): use 2x|jg|x|kg| decomposition
    
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2: 7 terms
    # For A (2x2), B (2x2), C (2x2):
    # C00 = M5 - M1 - M4 + M6
    # C01 = M1 - M5 + M4  
    # C10 = M2 - M6
    # C11 = M5 + M6 - M2 - M7
    # where:
    # M1 = (A00+A01)(B00+B11)
    # M2 = (A10+A11)(B00+B11)
    # M3 = A00(B00-B10)
    # M4 = A11(B01-B11)
    # M5 = (A00+A11)(B00+B10)
    # M6 = (A01+A10)(B00+B11)
    # M7 = (A01+A11)(B01-B11)
    
    # U terms: coefficients for A entries (i, j_local)
    # V terms: coefficients for B entries (j_local, k_local)
    # W terms: coefficients for C entries (k_local, i)
    
    strassen_7 = [
        # M1: (A00+A01)(B00+B11)
        ([(0,0,1.0),(0,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,-1.0),(0,1,1.0),(1,0,-1.0),(1,1,0.0)]),
        # M2: (A10+A11)(B00+B11)
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,0.0),(0,1,0.0),(1,0,1.0),(1,1,-1.0)]),
        # M3: A00(B00-B10)
        ([(0,0,1.0)], [(0,0,1.0),(1,0,-1.0)], [(0,0,1.0),(0,1,0.0),(1,0,0.0),(1,1,0.0)]),
        # M4: A11(B01-B11)
        ([(1,1,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,0,-1.0),(0,1,-1.0),(1,0,0.0),(1,1,0.0)]),
        # M5: (A00+A11)(B00+B10)
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,0,1.0)], [(0,0,1.0),(0,1,0.0),(1,0,0.0),(1,1,1.0)]),
        # M6: (A01+A10)(B00+B11)
        ([(0,1,1.0),(1,0,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(0,1,0.0),(1,0,-1.0),(1,1,1.0)]),
        # M7: (A01+A11)(B01-B11)
        ([(0,1,1.0),(1,1,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,0,0.0),(0,1,0.0),(1,0,0.0),(1,1,-1.0)]),
    ]
    
    # Wait, I need to double-check the W coefficients.
    # C00 = M5 - M1 - M4 + M6
    # C01 = M1 - M5 + M4
    # C10 = M2 - M6
    # C11 = M5 + M6 - M2 - M7
    
    # So W contributions:
    # M1: C00 -= 1, C01 += 1 => W[(0,0)] -= 1, W[(0,1)] += 1
    # M2: C10 += 1, C11 -= 1 => W[(1,0)] += 1, W[(1,1)] -= 1
    # M3: C00 += 1 => W[(0,0)] += 1
    # M4: C00 -= 1, C01 -= 1 => W[(0,0)] -= 1, W[(0,1)] -= 1
    # M5: C00 += 1, C11 += 1 => W[(0,0)] += 1, W[(1,1)] += 1
    # M6: C00 += 1, C10 -= 1, C11 += 1 => W[(0,0)] += 1, W[(1,0)] -= 1, W[(1,1)] += 1
    # M7: C11 -= 1 => W[(1,1)] -= 1
    
    strassen_7 = [
        # M1
        ([(0,0,1.0),(0,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        # M2
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(1,0,1.0),(1,1,-1.0)]),
        # M3
        ([(0,0,1.0)], [(0,0,1.0),(1,0,-1.0)], [(0,0,1.0)]),
        # M4
        ([(1,1,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,0,-1.0),(0,1,-1.0)]),
        # M5
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,0,1.0)], [(0,0,1.0),(1,1,1.0)]),
        # M6
        ([(0,1,1.0),(1,0,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,0,-1.0),(1,1,1.0)]),
        # M7
        ([(0,1,1.0),(1,1,1.0)], [(0,1,1.0),(1,1,-1.0)], [(1,1,-1.0)]),
    ]
    
    def add_2x2x2_strassen(i_base, j_base, k_base, col):
        for u_terms, v_terms, w_terms in strassen_7:
            for (li, lj, val) in u_terms:
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][col] += val
            for (lj, lk, val) in v_terms:
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][col] += val
            for (lk, li, val) in w_terms:
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][col] += val
            col += 1
        return col
    
    def add_2x1x2(i_base, j, k_base, col):
        # Naive 2x1x2: 4 terms
        for i in range(2):
            for k in range(2):
                U[(i_base + i) * m + j][col] = 1.0
                V[j * p + (k_base + k)][col] = 1.0
                W[(k_base + k) * n + (i_base + i)][col] = 1.0
                col += 1
        return col
    
    # Build the decomposition
    for s in range(3):
        k_base = s * 2
        col = add_2x2x2_strassen(0, 0, k_base, col)
        col = add_2x1x2(0, 2, k_base, col)
        col = add_2x2x2_strassen(0, 3, k_base, col)
    
    if time.time() - _START_TIME > _DEADLINE:
        raise TimeoutError("Exceeded time budget")
    
    # Verify the decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute tensor: T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
    # U shape: (n*m, rank) -> reshape to (n, m, rank)
    # V shape: (m*p, rank) -> reshape to (m, p, rank)  
    # W shape: (p*n, rank) -> reshape to (p, n, rank)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_t U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
    # This is a tensor contraction. Let's compute it.
    T = np.zeros((n, m, p))
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
    
    # Check: T[i,j,k] should be 1 for all i,j,k
    expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T - expected))
    
    if max_err > 1e-10:
        # Fallback: use naive rank-60 decomposition
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        if time.time() - _START_TIME > _DEADLINE:
            raise TimeoutError("Exceeded time budget")
        return {"U": U, "V": V, "W": W}
    
    if time.time() - _START_TIME > _DEADLINE:
        raise TimeoutError("Exceeded time budget")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
