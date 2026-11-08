import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8
    
    # Strassen 2x2x2 decomposition
    # For local indices i,j,k in {0,1}:
    # U local index: i*2+j
    # V local index: j*2+k
    # W local index: k*2+i
    
    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # Each entry is (local_index, value)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1)]),  # M2
        ([(0, 1)], [(2, 1), (3, -1)], [(2, 1)]),  # M3
        ([(3, 1)], [(1, 1), (2, -1)], [(0, 1), (1, 1)]),  # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (3, 1)]),  # M5
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(1, 1)]),  # M6
        ([(1, 1), (3, -1)], [(1, 1), (3, 1)], [(0, -1), (3, 1)]),  # M7
    ]
    
    # First, verify the 2x2x2 Strassen decomposition
    def verify_222():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for r in range(7):
                        u_val = 0
                        v_val = 0
                        w_val = 0
                        u_idx = i * 2 + j
                        v_idx = j * 2 + k
                        w_idx = k * 2 + i
                        for (idx, val) in strassen_terms[r][0]:
                            if idx == u_idx:
                                u_val += val
                        for (idx, val) in strassen_terms[r][1]:
                            if idx == v_idx:
                                v_val += val
                        for (idx, val) in strassen_terms[r][2]:
                            if idx == w_idx:
                                w_val += val
                        total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    if not verify_222():
        # Fallback to trivial rank 64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the full 2x4x8 decomposition using 8 Strassen 2x2x2 blocks
    # j split into {0,1} and {2,3} (2 groups)
    # k split into {0,1}, {2,3}, {4,5}, {6,7} (4 groups)
    # Total: 2 * 4 = 8 blocks, each rank 7, total rank 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    for j_block in range(2):  # j_block: 0 or 1, local j in {0,1} -> global j = j_block*2 + local_j
        for k_block in range(4):  # k_block: 0-3, local k in {0,1} -> global k = k_block*2 + local_k
            base_rank = (j_block * 4 + k_block) * 7
            for r in range(7):
                cr = base_rank + r
                # U: local (i, j) -> global (i, j_block*2 + j)
                for (idx, val) in strassen_terms[r][0]:
                    li, lj = divmod(idx, 2)
                    global_i = li
                    global_j = j_block * 2 + lj
                    U[global_i * m + global_j][cr] += val
                # V: local (j, k) -> global (j_block*2 + j, k_block*2 + k)
                for (idx, val) in strassen_terms[r][1]:
                    lj, lk = divmod(idx, 2)
                    global_j = j_block * 2 + lj
                    global_k = k_block * 2 + lk
                    V[global_j * p + global_k][cr] += val
                # W: local (k, i) -> global (k_block*2 + k, i)
                for (idx, val) in strassen_terms[r][2]:
                    lk, li = divmod(idx, 2)
                    global_k = k_block * 2 + lk
                    global_i = li
                    W[global_k * n + global_i][cr] += val
    
    # Final verification with random matrices
    def full_verify():
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.randint(-2, 3, size=(n, m))
            B = rng.randint(-2, 3, size=(m, p))
            C_expected = A @ B
            
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0
                    for j in range(m):
                        for r in range(rank):
                            total += A[i, j] * U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    C_computed[i, k] = total
            
            if not np.allclose(C_computed, C_expected):
                return False
        return True
    
    if not full_verify():
        # Fallback to trivial rank 64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
