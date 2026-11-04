import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Use Strassen-based tiling: split into 2x2 blocks
    # A is 4x4 -> 2x2 grid of 2x2 blocks
    # B is 4x6 -> 2x3 grid of 2x2 blocks
    # Each output 2x2 block C_ij = A_i1 * B_1j + A_i2 * B_2j
    # So we need 2 * 2 * 3 = 12 instances of 2x2x2 multiplication
    # Each uses Strassen rank 7, total rank = 12 * 7 = 84
    
    # For each output block (oi, ok) where oi in {0,1}, ok in {0,1,2}
    # and for each input row-block (oj) in {0,1}:
    #   Add Strassen decomposition for A[oi*2:(oi+1)*2, oj*2:(oj+1)*2] * B[oj*2:(oj+1)*2, ok*2:(ok+1)*2]
    
    for oi in range(2):
        for oj in range(2):
            for ok in range(3):
                _add_strassen_2x2(oi * 2, oj * 2, ok * 2, U_cols, V_cols, W_cols, n, m, p)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Verify correctness
    if _time_remaining() > 10.0:
        ok = _verify_decomposition(U_final, V_final, W_final, n, m, p)
        if not ok:
            # Fallback to schoolbook if verification fails
            U_cols2, V_cols2, W_cols2 = _schoolbook(n, m, p)
            rank2 = len(U_cols2)
            U_final = [[U_cols2[r][i] for r in range(rank2)] for i in range(n * m)]
            V_final = [[V_cols2[r][i] for r in range(rank2)] for i in range(m * p)]
            W_final = [[W_cols2[r][i] for r in range(rank2)] for i in range(p * n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}

def _add_strassen_2x2(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """
    Add Strassen decomposition for 2x2 * 2x2 block multiplication.
    oi: row offset for A (and C)
    oj: col offset for A (and row offset for B)
    ok: col offset for B (and col offset for C)
    """
    # Strassen 7 terms for 2x2 * 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each term r contributes to U, V, W with sparse patterns
    # U has entries at (i,j) positions in the 2x2 A block
    # V has entries at (j,k) positions in the 2x2 B block
    # W has entries at (k,i) positions in the 2x2 C block (note: W index is k*n+i)
    
    strassen_terms = [
        # (U_pattern, V_pattern, W_pattern)
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # M2: (A21+A22)B11 -> C21 += 1, C22 -= 1
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(0,1): 1, (1,1): -1}),
        # M3: A11(B21-B11) -> C12 += 1, C22 += 1
        ({(0,0): 1},
         {(1,0): 1, (0,0): -1},
         {(1,0): 1, (1,1): 1}),
        # M4: A22(B12-B22) -> C11 -= 1, C21 += 1
        ({(1,1): 1},
         {(0,1): 1, (1,1): -1},
         {(0,0): -1, (0,1): 1}),
        # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (1,0): 1}),
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # M7: (A12-A22)(B21+B22) -> C11 += 1
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    for u_pat, v_pat, w_pat in strassen_terms:
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (r, c), coeff in u_pat.items():
            i = oi + r
            j = oj + c
            if 0 <= i < n and 0 <= j < m:
                u_col[i * m + j] += coeff
                
        for (r, c), coeff in v_pat.items():
            j = oj + r
            k = ok + c
            if 0 <= j < m and 0 <= k < p:
                v_col[j * p + k] += coeff
                
        for (r, c), coeff in w_pat.items():
            i = oi + r
            k = ok + c
            if 0 <= i < n and 0 <= k < p:
                w_col[k * n + i] += coeff
                
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition: one term per (i, j, k) triple.
    Rank = n * m * p
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                
                u_col[i * m + j] = 1
                v_col[j * p + k] = 1
                w_col[k * n + i] = 1
                
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
    
    return U_cols, V_cols, W_cols

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-6:
                    return False
    return True
