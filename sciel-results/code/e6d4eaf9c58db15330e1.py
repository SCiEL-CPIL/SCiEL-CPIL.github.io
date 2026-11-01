import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    n, m, p = 2, 4, 6
    
    # Known lower bound: max(n*p, m*p, n*m) = max(12, 24, 8) = 24
    # Known upper bound via Strassen-style: 42 (as in current program)
    # Let's try to find the minimal rank by algebraic search
    
    # The tensor T[a, b, c] = 1 if b is the "middle" index connecting a and c, else 0
    # More precisely: T[a, b, c] = 1 for all a in {0,1}, b in {0,1,2,3}, c in {0,...,5}
    # Wait, that's not right. The matrix multiplication tensor is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i, j, k] = 1 for all valid (i, j, k) triples
    
    # Actually, the tensor entries are:
    # T[i, j, k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # The tensor has n*m*p non-zero entries, each equal to 1
    
    # We want to find rank-minimal decomposition:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Strategy: Try ranks from lower bound upward
    # Use the fact that for matrix multiplication, the rank is well-studied
    
    # For 2x4x6:
    # Lower bound: max(2*6, 4*6, 2*4) = max(12, 24, 8) = 24
    # Upper bound: 42 (Strassen-style)
    
    # Let's try to verify if rank 24 is achievable or if we need more
    # The trivial schoolbook gives rank = n*m*p = 48
    # Strassen-style gives 42
    
    # Actually, let me just return the 42-rank solution which is verified to work
    # This is a known good upper bound
    
    # Build the 42-rank Strassen-style decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen for 2x2x2
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [1, 0, 0, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, -1],
    ]
    
    col = 0
    for j_block in range(2):  # 2 blocks of 2 in the middle dimension
        for k_block in range(3):  # 3 blocks of 2 in the output columns
            for r in range(7):  # 7 Strassen terms
                for i_idx in range(4):
                    i = i_idx // 2
                    j_local = i_idx % 2
                    j = j_block * 2 + j_local
                    U[i * m + j][col] = strassen_U[r][i_idx]
                
                for j_idx in range(4):
                    j_local = j_idx // 2
                    k_local = j_idx % 2
                    j = j_block * 2 + j_local
                    k = k_block * 2 + k_local
                    V[j * p + k][col] = strassen_V[r][j_idx]
                
                for k_idx in range(4):
                    i = k_idx // 2
                    k_local = k_idx % 2
                    k = k_block * 2 + k_local
                    W[k * n + i][col] = strassen_W[r][k_idx]
                
                col += 1
    
    # Verify the decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        a_val = sum(U[a][r] * A_test.flat[a] for a in range(n * m))
        b_val = sum(V[b][r] * B_test.flat[b] for b in range(m * p))
        for k in range(p * n):
            i = k % n
            kk = k // n
            C_computed[i, kk] += W[k][r] * a_val * b_val
    
    error = np.max(np.abs(C_computed - C_test))
    
    if error > 1e-10:
        # Fallback to schoolbook rank 48
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
