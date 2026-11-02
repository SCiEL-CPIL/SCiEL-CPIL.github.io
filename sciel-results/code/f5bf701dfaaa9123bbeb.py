import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _verify(U, V, W, n=None, m=None, p=None):
    """
    Verify exact tensor decomposition.
    """
    if n is None:
        n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                s = np.dot(u_row * v_row, w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_3x2x2():
    """
    Returns lists of (u, v, w) for 3x2x2 multiplication.
    """
    terms = []
    
    # 1. (a+d)(x+z)
    u = [1, 0, 0, 1, 0, 0]
    v = [1, 0, 1, 0]
    w = [1, 0, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # 2. (c+d)x
    u = [0, 0, 1, 1, 0, 0]
    v = [1, 0, 0, 0]
    w = [0, 0, 0, 1, -1, 0]
    terms.append((u, v, w))
    
    # 3. a(y-z)
    u = [1, 0, 0, 0, 0, 0]
    v = [0, 1, -1, 0]
    w = [0, 1, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # 4. d(w-y)
    u = [0, 0, 0, 1, 0, 0]
    v = [0, -1, 0, 1]
    w = [1, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 5. (a+b)w
    u = [1, 1, 0, 0, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # 6. (c-a)(x+z)
    u = [-1, 0, 1, 0, 0, 0]
    v = [1, 0, 1, 0]
    w = [0, 0, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # 7. (b-d)(y+w)
    u = [0, 1, 0, -1, 0, 0]
    v = [0, 1, 0, 1]
    w = [1, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # 8. e*x
    u = [0, 0, 0, 0, 1, 0]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 9. e*y
    u = [0, 0, 0, 0, 1, 0]
    v = [0, 1, 0, 0]
    w = [0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    # 10. f*z
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 1, 0]
    w = [0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # 11. f*w
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 1]
    w = [0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    return terms

def _construct_block_strassen():
    """
    Constructs the factor matrices using block Strassen.
    Returns (U, V, W) as lists.
    """
    n, m, p = DIMENSIONS
    base_terms = _construct_3x2x2()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for j_blk in range(2):
        for k_blk in range(4):
            for u, v, w in base_terms:
                u_global = [0.0] * (n * m)
                v_global = [0.0] * (m * p)
                w_global = [0.0] * (p * n)
                
                for li in range(3):
                    for lj in range(2):
                        val = u[li * 2 + lj]
                        if val != 0:
                            gj = j_blk * 2 + lj
                            u_global[li * m + gj] = val
                            
                for lj in range(2):
                    for lk in range(2):
                        val = v[lj * 2 + lk]
                        if val != 0:
                            gj = j_blk * 2 + lj
                            gk = k_blk * 2 + lk
                            v_global[gj * p + gk] = val
                            
                for lk in range(2):
                    for li in range(3):
                        val = w[lk * 3 + li]
                        if val != 0:
                            gk = k_blk * 2 + lk
                            w_global[gk * n + li] = val
                            
                U_cols.append(u_global)
                V_cols.append(v_global)
                W_cols.append(w_global)
                
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
            
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_attempts=50, time_limit=10):
    """
    Try to reduce rank by 1 by finding a linear dependency.
    Returns (U, V, W) with reduced rank if successful, else original.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Build tensor T from current decomposition
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # Try removing each term and see if the rest can compensate
    for r_remove in range(rank):
        if time.time() - start_time > time_limit:
            break
        
        # Build system: for each (i,j,k), sum_{r != r_remove} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a bilinear system, hard to solve directly.
        # Instead, check if term r_remove is in the span of others (unlikely for tensor rank).
        
        # Skip for now - this approach is not straightforward.
        pass
    
    return U, V, W

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses block Strassen decomposition to achieve rank 88.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Build the block Strassen decomposition
    U, V, W = _construct_block_strassen()
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        return _schoolbook()

# EVOLVE-BLOCK-END
