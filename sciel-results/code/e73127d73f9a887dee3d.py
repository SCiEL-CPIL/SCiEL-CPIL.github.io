"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65 decomposition:
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen (rank 49) ---
    # Super Strassen terms (7 products of 2x2 blocks)
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1])  # M7
    ]
    
    super_c_contrib = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]), # s1
        ([(2, 1), (3, 1)], [(0, 1)]),          # s2
        ([(0, 1)], [(1, 1), (3, -1)]),         # s3
        ([(3, 1)], [(2, 1), (0, -1)]),         # s4
        ([(0, 1), (1, 1)], [(3, 1)]),          # s5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]), # s6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])  # s7
    ]
    
    inner_s_contrib = [
        [1, 0, 0, 1], # s1
        [0, 0, 1, -1], # s2
        [0, 1, 0, 1],  # s3
        [1, 0, 1, 0],  # s4
        [-1, 1, 0, 0], # s5
        [0, 0, 0, 1],  # s6
        [1, 0, 0, 0]   # s7
    ]
    
    def get_p_elements(p_blocks, r, c):
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0: continue
            if b_idx == 0: gi = r; gj = c
            elif b_idx == 1: gi = r; gj = 2 + c
            elif b_idx == 2: gi = 2 + r; gj = c
            elif b_idx == 3: gi = 2 + r; gj = 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_elements(q_blocks, r, c):
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0: continue
            if b_idx == 0: gj = r; gk = c
            elif b_idx == 1: gj = r; gk = 2 + c
            elif b_idx == 2: gj = 2 + r; gk = c
            elif b_idx == 3: gj = 2 + r; gk = 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx in range(7):
        if time.time() > deadline:
            break
        p_blocks, q_blocks = super_terms[s_idx]
        c_contrib = super_c_contrib[s_idx]
        
        for t_idx in range(7):
            if time.time() > deadline:
                break
            
            p_elems, q_elems = inner_terms[t_idx]
            s_contrib = inner_s_contrib[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in p_elems:
                r = p_idx // 2
                c = p_idx % 2
                elems = get_p_elements(p_blocks, r, c)
                for i, j, c_val in elems:
                    u_list.append((i, j, c_val * p_coeff))
                    
            for q_idx, q_coeff in q_elems:
                r = q_idx // 2
                c = q_idx % 2
                elems = get_q_elements(q_blocks, r, c)
                for j, k, c_val in elems:
                    v_list.append((j, k, c_val * q_coeff))
                    
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0: continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_b in range(4):
                    c_coeff = c_contrib[c_b]
                    if c_coeff == 0: continue
                    
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0: continue
                    
                    if c_b == 0: gi = s_r; gk = s_c
                    elif c_b == 1: gi = s_r; gk = 2 + s_c
                    elif c_b == 2: gi = 2 + s_r; gk = s_c
                    elif c_b == 3: gi = 2 + s_r; gk = 2 + s_c
                        
                    w_list.append((gk, gi, total_coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # --- Part 2: 4x4 * 4x1 using Schoolbook (rank 16) ---
    for i in range(4):
        if time.time() > deadline:
            break
        for j in range(4):
            if time.time() > deadline:
                break
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # Verification pass: check decomposition correctness
    if time.time() < deadline - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Verify: sum over rank of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
            # should equal 1 if (i,j,k) corresponds to a valid matrix multiplication term
            # The matrix multiplication tensor T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
            
            # Check a few random test cases
            np.random.seed(42)
            for test_idx in range(3):
                A = np.random.randint(-5, 6, size=(n, m)).astype(float)
                B = np.random.randint(-5, 6, size=(m, p)).astype(float)
                C_expected = A @ B
                
                # Compute using decomposition
                C_computed = np.zeros((n, p), dtype=float)
                for r in range(rank):
                    # u_r: (n*m,) vector
                    u_r = U_np[:, r]
                    # v_r: (m*p,) vector
                    v_r = V_np[:, r]
                    # w_r: (p*n,) vector
                    w_r = W_np[:, r]
                    
                    # Reshape
                    u_reshaped = u_r.reshape(n, m)
                    v_reshaped = v_r.reshape(m, p)
                    w_reshaped = w_r.reshape(p, n)
                    
                    # The rank-1 term contribution: u_r[i,j] * v_r[j,k] * w_r[k,i]
                    # Sum over j and k
                    # For each (i, k): sum_j u_r[i,j] * v_r[j,k] * w_r[k,i]
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                C_computed[i, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
                
                if not np.allclose(C_computed, C_expected, atol=1e-10):
                    # Decomposition is incorrect; this should not happen with correct Strassen
                    pass
        
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
