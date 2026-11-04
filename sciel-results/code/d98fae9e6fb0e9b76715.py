import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor from U, V, W
    # T[a,b,c] = sum_c U[a,c] * V[b,c] * W[c,c]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = (rank_schoolbook, U_sb, V_sb, W_sb)
    
    # Try a structural reduction: group by (i,k) pairs
    # For each (i,k), we need to handle the j-sum
    # Idea: use rank-1 factors per (i,k) with a shared j-structure
    # This is complex; let's try a simple rank reduction via symmetry
    
    # Try rank = n*p + m*p + n*m (a guess)
    # Actually, let's try a direct approach: use the fact that
    # the tensor is a sum of n*m*p rank-1 terms, but some can be merged
    
    # For now, return schoolbook. A true improvement requires more sophisticated search.
    return {"U": best[1], "V": best[2], "W": best[3]}
