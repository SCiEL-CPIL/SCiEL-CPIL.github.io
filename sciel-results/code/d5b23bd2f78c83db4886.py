import numpy as np
import time
import itertools

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    Strategy:
    1. M_{3,3,6} is isomorphic to 3 copies of M_{3,3,2} sharing the A factor.
    2. Rank(M_{3,3,6}) = 3 * Rank(M_{3,3,2}).
    3. We search for a low-rank decomposition of M_{3,3,2} with integer/half-integer entries.
    4. If found with rank R, we construct a rank 3R decomposition for M_{3,3,6}.
    5. Fallback to schoolbook rank 54.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Target: M_{3,3,2}
    n, m, p = 3, 3, 2
    dim_A = n * m  # 9
    dim_B = m * p  # 6
    dim_C = p * n  # 6
    
    T_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_nonzero[(a, b, c)] = 1.0

    def generate_schoolbook_3x3x2():
        rank = n * m * p  # 18
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return U, V, W

    best_U = None
    best_V = None
    best_W = None
    best_rank = 18  # Schoolbook

    # Try ranks from 14 to 17
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
            
        # Limit attempts per rank
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            U_rand = np.random.randint(-1, 2, size=(dim_A, target_rank))
            V_rand = np.random.randint(-1, 2, size=(dim_B, target_rank))
            
            W_rand = np.zeros((dim_C, target_rank))
            valid = True
            
            for c in range(dim_C):
                M_c = np.zeros((dim_A * dim_B, target_rank))
                b_c = np.zeros(dim_A * dim_B)
                
                idx = 0
                for a in range(dim_A):
                    for b in range(dim_B):
                        M_c[idx, :] = U_rand[a, :] * V_rand[b, :]
                        if (a, b, c) in T_nonzero:
                            b_c[idx] = 1.0
                        else:
                            b_c[idx] = 0.0
                        idx += 1
                
                try:
                    w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
                except:
                    valid = False
                    break
                
                w_doubled = w * 2
                closest_ints = np.round(w_doubled)
                error = np.max(np.abs(w_doubled - closest_ints))
                
                if error < 1e-9:
                    W_rand[c, :] = w
                else:
                    valid = False
                    break
            
            if valid:
                # Verify
                is_valid = True
                for (a, b, c), val in T_nonzero.items():
                    s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                    if abs(s - val) > 1e-9:
                        is_valid = False
                        break
                
                if is_valid:
                    # Check zero entries
                    for _ in range(20):
                        a = np.random.randint(0, dim_A)
                        b = np.random.randint(0, dim_B)
                        c = np.random.randint(0, dim_C)
                        if (a, b, c) not in T_nonzero:
                            s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                            if abs(s) > 1e-9:
                                is_valid = False
                                break
                    
                    if is_valid:
                        best_U = U_rand
                        best_V = V_rand
                        best_W = W_rand
                        best_rank = target_rank
                        break # Found valid for this rank, try lower rank? No, we iterate 14, 15... so 14 is best.
        
        if best_rank < 18:
            break # Found a rank < 18, stop searching higher ranks

    # Construct 3x3x6 solution
    if best_U is not None and best_rank < 18:
        total_rank = 3 * best_rank
        U6 = np.zeros((9, total_rank))
        V6 = np.zeros((18, total_rank))
        W6 = np.zeros((18, total_rank))
        
        # We have 3 blocks of B (columns 0-1, 2-3, 4-5)
        # And 3 blocks of C (columns 0-1, 2-3, 4-5)
        # The decomposition for M_{3,3,2} uses U, V, W.
        # For M_{3,3,6}, we repeat the U part, and place V, W in the corresponding blocks.
        
        for l in range(3):
            U6[:, l*best_rank:(l+1)*best_rank] = best_U
            # V block l corresponds to rows 0..2 of B, columns 2*l .. 2*l+1
            # In V6, row index is j*p + k.
            # For block l, k_local = 0,1. k_global = 2*l + k_local.
            # b_index = j*6 + k_global.
            # In best_V, b_index_local = j*2 + k_local.
            # So we map best_V[j*2+k_local, r] to V6[j*6 + 2*l + k_local, l*best_rank + r]
            
            for j in range(3):
                for k_local in range(2):
                    b_idx_local = j * 2 + k_local
                    b_idx_global = j * 6 + (2 * l + k_local)
                    V6[b_idx_global, l*best_rank:(l+1)*best_rank] = best_V[b_idx_local, :]
            
            # W block l corresponds to columns 2*l .. 2*l+1 of C.
            # In W6, row index is k*n + i.
            # For block l, k_local = 0,1. k_global = 2*l + k_local.
            # c_index = k_global*3 + i.
            # In best_W, c_index_local = k_local*3 + i.
            
            for k_local in range(2):
                for i in range(3):
                    c_idx_local = k_local * 3 + i
                    c_idx_global = (2 * l + k_local) * 3 + i
                    W6[c_idx_global, l*best_rank:(l+1)*best_rank] = best_W[c_idx_local, :]
        
        U_list = [[int(x) for x in row] for row in U6.tolist()]
        V_list = [[int(x) for x in row] for row in V6.tolist()]
        W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W6.tolist()]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook 54
        n, m, p = 3, 3, 6
        dim_A = 9
        dim_B = 18
        dim_C = 18
        rank = 54
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0

    # Target: M_{3,3,2}
    n, m, p = 3, 3, 2
    dim_A = n * m  # 9
    dim_B = m * p  # 6
    dim_C = p * n  # 6
    
    T_nonzero = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_nonzero[(a, b, c)] = 1.0

    best_U = None
    best_V = None
    best_W = None
    best_rank = 18  # Schoolbook

    # Try ranks from 14 to 17
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
            
        # Limit attempts per rank
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            U_rand = np.random.randint(-1, 2, size=(dim_A, target_rank))
            V_rand = np.random.randint(-1, 2, size=(dim_B, target_rank))
            
            W_rand = np.zeros((dim_C, target_rank))
            valid = True
            
            for c in range(dim_C):
                M_c = np.zeros((dim_A * dim_B, target_rank))
                b_c = np.zeros(dim_A * dim_B)
                
                idx = 0
                for a in range(dim_A):
                    for b in range(dim_B):
                        M_c[idx, :] = U_rand[a, :] * V_rand[b, :]
                        if (a, b, c) in T_nonzero:
                            b_c[idx] = 1.0
                        else:
                            b_c[idx] = 0.0
                        idx += 1
                
                try:
                    w, residuals, rank, s = np.linalg.lstsq(M_c, b_c, rcond=None)
                except:
                    valid = False
                    break
                
                w_doubled = w * 2
                closest_ints = np.round(w_doubled)
                error = np.max(np.abs(w_doubled - closest_ints))
                
                if error < 1e-9:
                    W_rand[c, :] = w
                else:
                    valid = False
                    break
            
            if valid:
                # Verify
                is_valid = True
                for (a, b, c), val in T_nonzero.items():
                    s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                    if abs(s - val) > 1e-9:
                        is_valid = False
                        break
                
                if is_valid:
                    # Check zero entries
                    for _ in range(20):
                        a = np.random.randint(0, dim_A)
                        b = np.random.randint(0, dim_B)
                        c = np.random.randint(0, dim_C)
                        if (a, b, c) not in T_nonzero:
                            s = np.sum(U_rand[a, :] * V_rand[b, :] * W_rand[c, :])
                            if abs(s) > 1e-9:
                                is_valid = False
                                break
                    
                    if is_valid:
                        best_U = U_rand
                        best_V = V_rand
                        best_W = W_rand
                        best_rank = target_rank
                        break # Found valid for this rank
        
        if best_rank < 18:
            break # Found a rank < 18, stop searching higher ranks

    # Construct 3x3x6 solution
    if best_U is not None and best_rank < 18:
        total_rank = 3 * best_rank
        U6 = np.zeros((9, total_rank))
        V6 = np.zeros((18, total_rank))
        W6 = np.zeros((18, total_rank))
        
        for l in range(3):
            U6[:, l*best_rank:(l+1)*best_rank] = best_U
            
            for j in range(3):
                for k_local in range(2):
                    b_idx_local = j * 2 + k_local
                    b_idx_global = j * 6 + (2 * l + k_local)
                    V6[b_idx_global, l*best_rank:(l+1)*best_rank] = best_V[b_idx_local, :]
            
            for k_local in range(2):
                for i in range(3):
                    c_idx_local = k_local * 3 + i
                    c_idx_global = (2 * l + k_local) * 3 + i
                    W6[c_idx_global, l*best_rank:(l+1)*best_rank] = best_W[c_idx_local, :]
        
        U_list = [[int(x) for x in row] for row in U6.tolist()]
        V_list = [[int(x) for x in row] for row in V6.tolist()]
        W_list = [[float(np.round(x * 2) / 2) for x in row] for row in W6.tolist()]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook 54
        n, m, p = 3, 3, 6
        dim_A = 9
        dim_B = 18
        dim_C = 18
        rank = 54
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
