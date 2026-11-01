import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    Falls back to schoolbook (rank 64) if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2
    # U: [A00, A01, A10, A11]
    # V: [B00, B01, B10, B11]
    # W: [C00, C01, C10, C11]
    
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [-1, 0, 1, 0], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1]   # M7: B10+B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+1, C11+1
        [0, 0, 1, -1], # M2: C10+1, C11-1
        [0, 1, 0, 1],  # M3: C01+1, C11+1
        [1, 0, 1, 0],  # M4: C00+1, C10+1
        [-1, 1, 0, 0], # M5: C00-1, C01+1
        [0, 0, 0, 1],  # M6: C11+1
        [1, 0, 0, 0]   # M7: C00+1
    ]
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for l in range(8):
        for s in range(7):
            col = 7 * l + s
            
            # U: same for all blocks
            for a_idx in range(4):
                U[a_idx, col] = strassen_U[s][a_idx]
            
            # V: local B indices mapped to global
            # B00 -> 2*l, B01 -> 2*l+1, B10 -> 16+2*l, B11 -> 16+2*l+1
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            for b_idx in range(4):
                V[b_rows[b_idx], col] = strassen_V[s][b_idx]
            
            # W: local C indices mapped to global
            # C00 -> 4*l, C01 -> 4*l+2, C10 -> 4*l+1, C11 -> 4*l+3
            c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
            for c_idx in range(4):
                W[c_rows[c_idx], col] = strassen_W[s][c_idx]
    
    # Verify efficiently using tensor contraction logic
    # T[u,v,w] = sum_s U[u,s] V[v,s] W[w,s]
    # We need T[i*m+j, j*p+k, k*n+i] == 1 for all i,j,k
    # And T[u,v,w] == 0 otherwise.
    
    # For speed, we can check the non-zero terms first, which are the ones that MUST be 1.
    # If those pass, we do a spot check or full check if time permits.
    # Given the small dimensions (4x32x32 tensor space, but sparse), a full check is fast enough.
    
    def verify(U, V, W, n, m, p):
        # Check required 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check spurious 0s (sample or full if fast)
        # Total entries: 4 * 32 * 32 = 4096.
        # This is very fast.
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
        
        # We can vectorize the zero check partially or just loop.
        # Looping 4096 items is trivial.
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) not in valid_set:
                        val = np.sum(U[u, :] * V[v, :] * W[w, :])
                        if abs(val) > 1e-9:
                            return False
        return True

    if time.time() < deadline:
        if verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
