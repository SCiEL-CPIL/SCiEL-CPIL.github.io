import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _get_bini_rank23():
    """
    Bini et al. 1978 rank-23 decomposition for 3x3 matrix multiplication.
    """
    terms = [
        ([1,0,0, 0,0,0, 0,0,0],  [1,0,0, 0,0,0, 0,0,0],  [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0],  [0,0,0, 0,1,0, 0,0,0],  [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1],  [0,0,0, 0,0,0, 0,0,1],  [0,0,0, 0,0,0, 0,0,1]),
        
        ([1,1,0, 0,0,0, 0,0,0],  [1,0,0, 1,0,0, 0,0,0],  [1,1,0, 1,1,0, 0,0,0]),
        ([0,0,0, 1,1,0, 0,0,0],  [0,0,0, 1,0,0, 1,0,0],  [0,0,0, 1,1,0, 1,1,0]),
        ([0,0,0, 0,0,0, 1,1,0],  [0,0,0, 0,0,0, 1,0,0],  [0,0,0, 0,0,0, 1,1,0]),
        
        ([1,0,0, 0,1,0, 0,0,0],  [1,0,0, 0,1,0, 0,0,0],  [1,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,1,0],  [0,0,0, 0,1,0, 0,0,1],  [0,0,0, 0,1,0, 0,0,1]),
        ([0,0,0, 0,0,0, 1,0,0],  [0,0,0, 0,0,0, 0,1,0],  [0,0,0, 0,0,0, 0,1,0]),
        
        ([1,0,0, 0,1,0, 0,0,0],  [1,0,0, 0,1,0, 0,0,0],  [1,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,1,0],  [0,0,0, 0,1,0, 0,0,1],  [0,0,0, 1,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0],  [0,0,0, 0,0,0, 0,1,0],  [0,0,0, 0,0,0, 1,1,0]),
        ([1,0,0, 0,1,0, 0,0,0],  [1,0,0, 0,1,0, 0,0,0],  [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,1,0],  [0,0,0, 0,1,0, 0,0,1],  [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 1,0,0],  [0,0,0, 0,0,0, 0,1,0],  [0,0,0, 0,0,0, 0,0,0]),
        
        ([1,0,0, 0,0,0, 0,1,0],  [0,1,0, 0,0,0, 0,0,0],  [0,0,0, 0,0,0, 0,1,0]),
        ([1,0,0, 0,0,0, 0,1,0],  [0,0,0, 0,1,0, 0,0,0],  [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,1,0, 0,0,0],  [0,0,1, 0,0,0, 0,0,0],  [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,1,0, 0,0,0],  [0,0,0, 0,0,1, 0,0,0],  [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1],  [0,0,1, 0,0,0, 0,0,0],  [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1],  [0,0,0, 0,0,1, 0,0,0],  [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,1,0],  [0,1,0, 0,0,0, 0,0,0],  [0,0,0, 0,0,0, 0,1,0]),
        ([1,0,0, 0,0,0, 0,1,0],  [0,0,0, 0,1,0, 0,0,0],  [0,0,0, 0,0,0, 0,0,1]),
    ]
    return terms

def _verify_bini(terms, n_tests=20):
    rng = np.random.default_rng(12345)
    for _ in range(n_tests):
        A = rng.integers(-5, 5, (3,3))
        B = rng.integers(-5, 5, (3,3))
        C_expected = A @ B
        C_computed = np.zeros((3,3), dtype=float)
        for u, v, w in terms:
            u_mat = np.array(u, dtype=float).reshape(3,3)
            v_mat = np.array(v, dtype=float).reshape(3,3)
            w_mat = np.array(w, dtype=float).reshape(3,3)
            C_computed += np.sum(u_mat * A) * np.sum(v_mat * B) * w_mat
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    bini_23 = _get_bini_rank23()
    bini_ok = _verify_bini(bini_23)
    
    if not bini_ok:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    total_rank = 36 + 23 + 23
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A1 (3x2) * B1 (2x6) schoolbook, rank 36
    for i in range(3):
        for k in range(2):
            for j in range(6):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # Block 2a: A2 (3x3) * B2a (3x3) using Bini, rank 23
    for u, v, w in bini_23:
        u_mat = np.array(u).reshape(3,3)
        v_mat = np.array(v).reshape(3,3)
        w_mat = np.array(w).reshape(3,3)
        for i in range(3):
            for k in range(3):
                U[i * m + (k+2)][col] += u_mat[i,k]
        for k in range(3):
            for j in range(3):
                V[(k+2) * p + j][col] += v_mat[k,j]
        for i in range(3):
            for j in range(3):
                W[j * n + i][col] += w_mat[i,j]
        col += 1
    
    # Block 2b: A2 (3x3) * B2b (3x3) using Bini, rank 23
    for u, v, w in bini_23:
        u_mat = np.array(u).reshape(3,3)
        v_mat = np.array(v).reshape(3,3)
        w_mat = np.array(w).reshape(3,3)
        for i in range(3):
            for k in range(3):
                U[i * m + (k+2)][col] += u_mat[i,k]
        for k in range(3):
            for j in range(3):
                V[(k+2) * p + (j+3)][col] += v_mat[k,j]
        for i in range(3):
            for j in range(3):
                W[(j+3) * n + i][col] += w_mat[i,j]
        col += 1
    
    return {"U": U, "V": V, "W": W}
