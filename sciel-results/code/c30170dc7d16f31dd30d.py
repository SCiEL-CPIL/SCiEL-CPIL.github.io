import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = 3, 4, 4
    
    # Local 3x2x2 decomposition rank 11
    local_terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
        ([ (4, 1) ], [ (0, 1) ], [ (4, 1) ]),
        ([ (5, 1) ], [ (2, 1) ], [ (4, 1) ]),
        ([ (4, 1) ], [ (1, 1) ], [ (5, 1) ]),
        ([ (5, 1) ], [ (3, 1) ], [ (5, 1) ]),
    ]
    
    # Verify the local decomposition is correct
    def verify_local():
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        for l_idx, (u_list, v_list, w_list) in enumerate(local_terms):
            for idx, val in u_list:
                U_loc[idx, l_idx] += val
            for idx, val in v_list:
                V_loc[idx, l_idx] += val
            for idx, val in w_list:
                W_loc[idx, l_idx] += val
        
        # Check: sum_k U[:,k] * V[:,k] * W[:,k] should give the 3x2x2 tensor
        # Tensor T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
        # For 3x2x2: a in 0..5 (r*2+c), b in 0..3 (r*2+c), c in 0..5 (r*2+c)
        # C[r,c] = sum_i A[r,i] * B[i,c]
        # T[a,b,c] = 1 if a corresponds to (r,i), b to (i,c), c to (r,c)
        
        for a in range(6):
            r_a, c_a = a // 2, a % 2
            for b in range(4):
                r_b, c_b = b // 2, b % 2
                # A[r_a, c_a] * B[r_b, c_b] contributes to C[r_a, c_b] if c_a == r_b
                if c_a == r_b:
                    c = r_a * 2 + c_b
                    expected = 1.0
                    actual = sum(U_loc[a, k] * V_loc[b, k] * W_loc[c, k] for k in range(11))
                    if abs(actual - expected) > 1e-10:
                        return False
                else:
                    for c in range(6):
                        actual = sum(U_loc[a, k] * V_loc[b, k] * W_loc[c, k] for k in range(11))
                        if abs(actual) > 1e-10:
                            return False
        return True
    
    if not verify_local():
        # Fallback: use trivial decomposition for local block
        local_terms = []
        for a in range(6):
            for b in range(4):
                r_a, c_a = a // 2, a % 2
                r_b, c_b = b // 2, b % 2
                if c_a == r_b:
                    c = r_a * 2 + c_b
                    local_terms.append(([(a, 1)], [(b, 1)], [(c, 1)]))
        # This gives rank 12 for local, total 48
    
    local_rank = len(local_terms)
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    rank = 4 * local_rank
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for b_idx, (a_cs, b_rs, b_cs, c_cs) in enumerate(blocks):
        col_offset = b_idx * local_rank
        for l_idx, (u_list, v_list, w_list) in enumerate(local_terms):
            col = col_offset + l_idx
            for idx, val in u_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_a_idx = r_loc * m + (a_cs + c_loc)
                U[g_a_idx, col] += val
            for idx, val in v_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_b_idx = (b_rs + r_loc) * p + (b_cs + c_loc)
                V[g_b_idx, col] += val
            for idx, val in w_list:
                r_loc = idx // 2
                c_loc = idx % 2
                g_c_idx = (c_cs + c_loc) * n + r_loc
                W[g_c_idx, col] += val
        
        if time.time() > deadline:
            break
    
    # Verify full decomposition
    def verify_full():
        for r in range(n):
            for c in range(p):
                for i in range(m):
                    a_idx = r * m + i
                    b_idx = i * p + c
                    w_idx = c * n + r
                    expected = 1.0
                    actual = sum(U[a_idx, k] * V[b_idx, k] * W[w_idx, k] for k in range(rank))
                    if abs(actual - expected) > 1e-10:
                        return False
        # Check zeros
        for r in range(n):
            for c in range(p):
                for i in range(m):
                    for j in range(m):
                        if i != j:
                            a_idx = r * m + j
                            b_idx = i * p + c
                            w_idx = c * n + r
                            actual = sum(U[a_idx, k] * V[b_idx, k] * W[w_idx, k] for k in range(rank))
                            if abs(actual) > 1e-10:
                                return False
        return True
    
    if not verify_full():
        # Fallback to trivial rank 48
        rank = 48
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r in range(n):
            for i in range(m):
                for c in range(p):
                    k = r * m * p + i * p + c
                    U[r * m + i, k] = 1.0
                    V[i * p + c, k] = 1.0
                    W[c * n + r, k] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
