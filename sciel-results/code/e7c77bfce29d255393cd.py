import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: ensure we finish well before the 120s/150s limit.
    # This construction is deterministic and fast, but we add a safety check.
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    n, m, p = DIMENSIONS
    
    # Pre-compute Strassen 2x2 factors
    # u, v, w are 4x7 matrices
    # Indices 0..3 correspond to 00, 01, 10, 11
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    # Define the 7 Strassen 4x4 terms
    # Each term is defined by coefficients for A blocks, B blocks, C blocks.
    # Block indices (br, bc) where br, bc in {0, 1}.
    # A block (br, bc) maps to A[2*br:2*br+2, 2*bc:2*bc+2]
    # B block (br, bc) maps to B[2*br:2*br+2, 2*bc:2*bc+2]
    # C block (br, bc) maps to C[2*br:2*br+2, 2*bc:2*bc+2]
    
    # M1 = (A11+A22)(B11+B22)
    # A: {(0,0):1, (1,1):1}
    # B: {(0,0):1, (1,1):1}
    # C: {(0,0):1, (1,1):1}
    
    # M2 = (A21+A22)B11
    # A: {(1,0):1, (1,1):1}
    # B: {(0,0):1}
    # C: {(1,0):1, (1,1):-1}
    
    # M3 = A11(B12-B22)
    # A: {(0,0):1}
    # B: {(0,1):1, (1,1):-1}
    # C: {(0,1):1, (1,1):1}
    
    # M4 = A22(B21-B11)
    # A: {(1,1):1}
    # B: {(1,0):1, (0,0):-1}
    # C: {(0,0):1, (1,0):1}
    
    # M5 = (A11+A12)B22
    # A: {(0,0):1, (0,1):1}
    # B: {(1,1):1}
    # C: {(0,0):-1, (0,1):1}
    
    # M6 = (A21-A11)(B11+B12)
    # A: {(1,0):1, (0,0):-1}
    # B: {(0,0):1, (0,1):1}
    # C: {(1,1):1}
    
    # M7 = (A12-A22)(B21+B22)
    # A: {(0,1):1, (1,1):-1}
    # B: {(1,0):1, (1,1):1}
    # C: {(0,0):1}
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]
    
    # Rank calculation
    rank_4x4x4 = 7 * 7 # 49
    rank_4x4x1 = 4 * 4 # 16
    rank_4x1x4 = 4 * 4 # 16
    rank_4x1x1 = 4 * 1 # 4
    total_rank = rank_4x4x4 + rank_4x4x1 + rank_4x1x4 + rank_4x1x1 # 85
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(a_coeffs, b_coeffs, c_coeffs, start_col):
        # a_coeffs, b_coeffs, c_coeffs: dict (br, bc) -> coeff
        for k in range(7):
            u_vec = s_u[:, k] # 4 entries
            v_vec = s_v[:, k]
            w_vec = s_w[:, k]
            
            c_idx = start_col + k
            
            # Distribute U
            for (a_br, a_bc), a_coeff in a_coeffs.items():
                if a_coeff == 0: continue
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            U[global_i * 5 + global_j, c_idx] += a_coeff * u_val
            
            # Distribute V
            for (b_br, b_bc), b_coeff in b_coeffs.items():
                if b_coeff == 0: continue
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            V[global_j * 5 + global_k, c_idx] += b_coeff * v_val
            
            # Distribute W
            for (c_br, c_bc), c_coeff in c_coeffs.items():
                if c_coeff == 0: continue
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_i = 2 * c_br + lr
                            global_k = 2 * c_bc + lc
                            W[global_k * 4 + global_i, c_idx] += c_coeff * w_val

    # Add 4x4x4 terms
    for t_idx, (a_c, b_c, c_c) in enumerate(terms):
        add_term(a_c, b_c, c_c, col)
        col += 7
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Add 4x4x1 terms (A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4])
    # C index: k=4, i in 0..3. W row = 4*4 + i.
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    # Add 4x1x4 terms (A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4])
    # C index: k in 0..3, i in 0..3. W row = k*4 + i.
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            
    # Add 4x1x1 terms (A[0:4, 4] * B[4, 4] -> C[0:4, 4])
    # C index: k=4, i in 0..3. W row = 4*4 + i.
    for i in range(4):
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
