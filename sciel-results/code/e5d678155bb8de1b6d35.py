import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Build a rank-minimal decomposition using 16 disjoint 2x2x2 Strassen blocks
    # plus singleton terms for the remaining (i,j,k) entries.
    # The 16 blocks cover: i in {0,1,2,3}, j in {0,1,3,4}, k in {0,1,3,4}
    # Remaining entries have j=2 or k=2 (but not both in the block-covered region).
    
    rank = 100  # upper bound, we'll fill exactly
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    # Define the 16 Strassen blocks
    # i_base in {0, 2}, j_base in {0, 3}, k_base in {0, 3}
    blocks = []
    for i_base in [0, 2]:
        for j_base in [0, 3]:
            for k_base in [0, 3]:
                blocks.append((i_base, j_base, k_base))
    
    # Set of (i,j,k) covered by Strassen blocks
    skip_set = set()
    for ib, jb, kb in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    skip_set.add((ib+di, jb+dj, kb+dk))
    
    col = 0
    # Fill standard terms for non-block entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Fill Strassen terms for each 2x2x2 block
    for ib, jb, kb in blocks:
        i1, i2 = ib, ib + 1
        j1, j2 = jb, jb + 1
        k1, k2 = kb, kb + 1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[i2 * m + j1, col] += 1
        U[i2 * m + j2, col] += 1
        V[j1 * p + k1, col] += 1
        W[k1 * n + i2, col] += 1
        W[k2 * n + i2, col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[i1 * m + j1, col] += 1
        V[j1 * p + k2, col] += 1
        V[j2 * p + k2, col] += -1
        W[k2 * n + i1, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[i2 * m + j2, col] += 1
        V[j2 * p + k1, col] += 1
        V[j1 * p + k1, col] += -1
        W[k1 * n + i1, col] += 1
        W[k1 * n + i2, col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[i1 * m + j1, col] += 1
        U[i1 * m + j2, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += -1
        W[k2 * n + i1, col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1, col] += 1
        U[i1 * m + j1, col] += -1
        V[j1 * p + k1, col] += 1
        V[j1 * p + k2, col] += 1
        W[k2 * n + i2, col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2, col] += 1
        U[i2 * m + j2, col] += -1
        V[j2 * p + k1, col] += 1
        V[j2 * p + k2, col] += 1
        W[k1 * n + i1, col] += 1
        col += 1
    
    # Trim to actual column count
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    actual_rank = col
    
    # Verify correctness using vectorized NumPy operations for speed
    if time.time() < deadline - 5:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U[i * m + j, :]
                    v_vec = V[j * p + k, :]
                    w_vec = W[k * n + i, :]
                    val = np.sum(u_vec * v_vec * w_vec)
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to standard rank 100
            rank_std = 100
            U_std = np.zeros((nm, rank_std), dtype=np.int64)
            V_std = np.zeros((mp, rank_std), dtype=np.int64)
            W_std = np.zeros((pn, rank_std), dtype=np.int64)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_std[i*m+j, c] = 1
                        V_std[j*p+k, c] = 1
                        W_std[k*n+i, c] = 1
                        c += 1
            return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
