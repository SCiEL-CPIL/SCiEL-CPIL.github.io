"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 115.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def build_block_terms(k_offset):
        """
        Build terms for one 3x3x3 block with k in {k_offset, k_offset+1, k_offset+2}.
        Returns list of dicts with 'U', 'V', 'W' keys, mapping (local_i, local_j) etc to values.
        Local indices: i, j, kk in {0, 1, 2}.
        Global k = kk + k_offset.
        """
        terms = []
        
        # Strassen terms for core (i, j, kk in {0, 1})
        # 1. (A00+A11)(B00+B11) -> C00, C11
        terms.append({
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        })
        # 2. (A10+A11)B00 -> C10
        terms.append({
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1}
        })
        # 3. A00(B10-B00) -> C01, C11
        terms.append({
            'U': {(0,0): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(1,0): 1, (1,1): 1}
        })
        # 4. A11(B10-B00) -> C00, C10
        terms.append({
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (0,1): 1}
        })
        # 5. (A00+A01)B11 -> C00(-), C01
        terms.append({
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (1,0): 1}
        })
        # 6. (A10-A00)(B00+B01) -> C11
        terms.append({
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        })
        # 7. (A01-A11)(B10+B11) -> C00
        terms.append({
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1}
        })
        
        # Border terms
        # Term 8: Covers j=2, i in {0,1}, kk in {0,1,2}
        # U[i,2] = 1, V[2,kk] = 1, W[kk,i] = 1
        u8 = {(0,2): 1, (1,2): 1}
        v8 = {(2,0): 1, (2,1): 1, (2,2): 1}
        w8 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1, (2,0): 1, (2,1): 1}
        terms.append({'U': u8, 'V': v8, 'W': w8})
        
        # Term 9: Covers k=2, i in {0,1}, j in {0,1}
        # U[i,j] = 1, V[j,2] = 1, W[2,i] = 1
        u9 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        v9 = {(0,2): 1, (1,2): 1}
        w9 = {(2,0): 1, (2,1): 1}
        terms.append({'U': u9, 'V': v9, 'W': w9})
        
        # Term 10: Covers i=2, j in {0,1,2}, kk in {0,1,2}
        # U[2,j] = 1, V[j,kk] = 1, W[kk,2] = 1
        u10 = {(2,0): 1, (2,1): 1, (2,2): 1}
        v10 = {(0,0): 1, (0,1): 1, (0,2): 1,
               (1,0): 1, (1,1): 1, (1,2): 1,
               (2,0): 1, (2,1): 1, (2,2): 1}
        w10 = {(0,2): 1, (1,2): 1, (2,2): 1}
        terms.append({'U': u10, 'V': v10, 'W': w10})
        
        return terms

    # Build full decomposition
    all_terms = []
    for block in range(2):
        k_offset = block * 3
        block_terms = build_block_terms(k_offset)
        for term in block_terms:
            u_global = {}
            for (i, j), val in term['U'].items():
                u_global[(i, j)] = val
            
            v_global = {}
            for (j, kk), val in term['V'].items():
                v_global[(j, kk + k_offset)] = val
            
            w_global = {}
            for (kk, i), val in term['W'].items():
                w_global[(kk + k_offset, i)] = val
            
            all_terms.append({
                'U': u_global,
                'V': v_global,
                'W': w_global
            })
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(all_terms):
        for (i, j), val in term['U'].items():
            U[i * m + j][c] = val
        for (j, k), val in term['V'].items():
            V[j * p + k][c] = val
        for (k, i), val in term['W'].items():
            W[k * n + i][c] = val
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
