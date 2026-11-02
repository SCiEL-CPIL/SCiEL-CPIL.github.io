"""Tensor decomposition for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-10):
    """Check that sum_r U[:,r] V[:,r] W[:,r] equals the matrix multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', Ua[:, r], Va[:, r], Wa[:, r])
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    if np.max(np.abs(T - T_target)) > tol:
        return False
    return True

def _get_rank26_solution():
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1
    
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[0][2] = -1
    
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1
    
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    U[6][7] = 1; V[0][7] = 1; W[2][7] = 1
    U[7][8] = 1; V[3][8] = 1; W[2][8] = 1
    U[8][9] = 1; V[6][9] = 1; W[2][9] = 1
    
    U[6][10] = 1; V[1][10] = 1; W[5][10] = 1
    U[7][11] = 1; V[4][11] = 1; W[5][11] = 1
    U[8][12] = 1; V[7][12] = 1; W[5][12] = 1
    
    U[6][13] = 1; V[2][13] = 1; W[8][13] = 1
    U[7][14] = 1; V[5][14] = 1; W[8][14] = 1
    U[8][15] = 1; V[8][15] = 1; W[8][15] = 1
    
    U[0][16] = 1; V[2][16] = 1; W[6][16] = 1
    U[1][17] = 1; V[5][17] = 1; W[6][17] = 1
    U[2][18] = 1; V[8][18] = 1; W[6][18] = 1
    
    U[3][19] = 1; V[2][19] = 1; W[7][19] = 1
    U[4][20] = 1; V[5][20] = 1; W[7][20] = 1
    U[5][21] = 1; V[8][21] = 1; W[7][21] = 1
    
    U[2][22] = 1; V[6][22] = 1; W[0][22] = 1
    U[2][23] = 1; V[7][23] = 1; W[1][23] = 1
    U[5][24] = 1; V[6][24] = 1; W[3][24] = 1
    U[5][25] = 1; V[7][25] = 1; W[4][25] = 1
    
    return U, V, W

def _try_random_rank25(start_time, deadline):
    """Try to find a rank 25 decomposition by random search."""
    n, m, p = 3, 3, 3
    rank = 25
    U0, V0, W0 = _get_rank26_solution()
    
    # We have 26 terms. We want to see if we can drop one and adjust the rest.
    # Or just random search.
    # Random search is expensive.
    # Let's try a simple heuristic: check if any column is linearly dependent?
    # Unlikely.
    
    # If we have time, do a few random restarts.
    # But 150s is short for a full random search of tensor rank.
    # We will just return None if we don't find it quickly.
    
    # For now, return None to fall back to rank 26.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find rank 25
    res25 = _try_random_rank25(start_time, deadline)
    if res25 is not None:
        return res25
        
    # Fall back to rank 26
    U, V, W = _get_rank26_solution()
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: brute-force rank 27
    rank27 = 27
    U27 = [[0.0] * rank27 for _ in range(n * m)]
    V27 = [[0.0] * rank27 for _ in range(m * p)]
    W27 = [[0.0] * rank27 for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i * m + j][r] = 1.0
                V27[j * p + k][r] = 1.0
                W27[k * n + i][r] = 1.0
                r += 1
    
    if _verify(U27, V27, W27):
        return {"U": U27, "V": V27, "W": W27}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
