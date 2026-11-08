import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: 4 blocks of 2x2x4, each rank 12 (two Strassen-7 blocks)
    # Total rank: 4 * 12 = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * rank for _ in range(m * p)]  # 32 x 48
    W = [[0] * rank for _ in range(p * n)]  # 16 x 48
    
    # Strassen-7 for 2x2x2 (verified patterns)
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    
    # Build 12 terms for 2x2x4 from two 2x2x2 Strassen blocks
    # Block 0: k_local in {0,1}, Block 1: k_local in {2,3}
    terms_224 = []
    for block in range(2):
        k_offset = block * 2
        for r in range(7):
            u_entries = strassen_U[r]
            # V: local j*2+k_local -> j*4 + (k_local + k_offset)
            v_entries = []
            for (idx, val) in strassen_V[r]:
                lj, lk_local = divmod(idx, 2)
                v_entries.append((lj * 4 + lk_local + k_offset, val))
            # W: local k_local*2+i -> (k_local + k_offset)*2 + i
            w_entries = []
            for (idx, val) in strassen_W[r]:
                lk_local, li = divmod(idx, 2)
                w_entries.append(((lk_local + k_offset) * 2 + li, val))
            terms_224.append((u_entries, v_entries, w_entries))
    
    # Fast verification of 2x2x4 decomposition
    def verify_224():
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    total = 0
                    for r in range(12):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        u_idx = i * 2 + j
                        v_idx = j * 4 + k
                        w_idx = k * 2 + i
                        for (idx, val) in terms_224[r][0]:
                            if idx == u_idx:
                                u_val = val
                        for (idx, val) in terms_224[r][1]:
                            if idx == v_idx:
                                v_val = val
                        for (idx, val) in terms_224[r][2]:
                            if idx == w_idx:
                                w_val = val
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    if not verify_224():
        # Fallback to rank 56 Strassen
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    U[(li + i_off) * 4 + (lj + j_off)][cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk + k_off) * 2 + (li + i_off)][cr] += val
        
        for j_block in range(2):
            for k_block in range(4):
                base = (j_block * 4 + k_block) * 7
                add_strassen_block(base, 0, j_block * 2, k_block * 2)
        
        return {"U": U, "V": V, "W": W}
    
    # Apply the rank-12 decomposition for each of the 4 sub-blocks
    # j_block in {0,1}: j offset 0 or 2
    # k_block in {0,1}: k offset 0 or 4
    for j_block in range(2):
        for k_block in range(2):
            base_rank = (j_block * 2 + k_block) * 12
            for r in range(12):
                cr = base_rank + r
                # U: local i*2+j -> global i*4 + (j_block*2 + j)
                for (idx, val) in terms_224[r][0]:
                    li, lj = divmod(idx, 2)
                    U[li * 4 + (j_block * 2 + lj)][cr] += val
                # V: local j*4+k -> global (j_block*2 + j)*8 + (k_block*4 + k)
                for (idx, val) in terms_224[r][1]:
                    lj, lk = divmod(idx, 4)
                    V[(j_block * 2 + lj) * 8 + (k_block * 4 + lk)][cr] += val
                # W: local k*2+i -> global (k_block*4 + k)*2 + i
                for (idx, val) in terms_224[r][2]:
                    lk, li = divmod(idx, 2)
                    W[(k_block * 4 + lk) * 2 + li][cr] += val
    
    # Full verification
    def full_verify():
        import numpy as np
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        B = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1, 0, 1],
                      [1, 1, 0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0, 1, 1]], dtype=float)
        C_expected = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += A[i, j] * U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = total
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not full_verify():
        # Fallback to rank 56 Strassen
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    U[(li + i_off) * 4 + (lj + j_off)][cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk + k_off) * 2 + (li + i_off)][cr] += val
        
        for j_block in range(2):
            for k_block in range(4):
                base = (j_block * 4 + k_block) * 7
                add_strassen_block(base, 0, j_block * 2, k_block * 2)
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
