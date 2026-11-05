import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build a decomposition of rank 85
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Part 2: 4x4x1 Schoolbook (Rank 16)
    # Part 3: 4x1x5 Schoolbook (Rank 20)
    
    rank_total = 85
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    # 2x2x2 Strassen factors: U, V, W each 4x7
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
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
    
    # Precompute sparse representations of Strassen vectors for speed
    s_u_sparse = []
    s_v_sparse = []
    s_w_sparse = []
    for k in range(7):
        s_u_sparse.append([(idx, val) for idx, val in enumerate(s_u[:, k]) if val != 0])
        s_v_sparse.append([(idx, val) for idx, val in enumerate(s_v[:, k]) if val != 0])
        s_w_sparse.append([(idx, val) for idx, val in enumerate(s_w[:, k]) if val != 0])
    
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        """
        Add a 4x4x4 Strassen term using 2x2x2 Strassen sub-factors.
        a_blocks: dict {(br,bc): coeff} - linear combo of A's 2x2 blocks
        b_blocks: dict {(br,bc): coeff} - linear combo of B's 2x2 blocks  
        c_blocks: dict {(br,bc): coeff} - linear combo of C's 2x2 blocks
        """
        for inner_k in range(7):
            col = col_start + inner_k
            
            # Distribute u_vec to A entries
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for local_idx, u_val in s_u_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * a_br + lr
                    gj = 2 * a_bc + lc
                    U[gi * 5 + gj, col] += a_coeff * u_val
            
            # Distribute v_vec to B entries
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for local_idx, v_val in s_v_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gj = 2 * b_br + lr
                    gk = 2 * b_bc + lc
                    V[gj * 5 + gk, col] += b_coeff * v_val
            
            # Distribute w_vec to C entries
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for local_idx, w_val in s_w_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * c_br + lr
                    gk = 2 * c_bc + lc
                    W[gk * 4 + gi, col] += c_coeff * w_val
    
    # 7 Strassen 4x4 terms for the 4x4x4 core
    # These correspond to the standard 2x2 Strassen formulas applied to 2x2 blocks of the 4x4 matrices
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),           # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),           # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),           # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),           # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),           # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),           # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() - start_time > deadline:
            break
        add_4x4_term(a_coeff, b_coeff, c_coeff, t_idx * 7)
    
    col = 49
    
    # 4x4x1 part: A[:, 0:4] * B[:, 4] -> C[:, 4]
    # A is 4x5, B is 5x5. 
    # Indices: A[i, j], B[j, k], C[i, k]
    # We want to compute C[i, 4] for i in 0..3.
    # C[i, 4] = sum_{j=0}^4 A[i, j] B[j, 4]
    # We have already handled j=0..3 in the 4x4x4 block? 
    # Wait, the 4x4x4 block handles A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4].
    # So we need to handle:
    # 1. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4] (Contribution to the 4x4 block of C)
    # 2. A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4] (Contribution to the 4x1 block of C)
    # 3. A[0:4, 4] * B[4, 4] -> C[0:4, 4] (Contribution to the 4x1 block of C)
    
    # Let's re-verify the block structure.
    # A is 4x5. Split into A_left (4x4) and A_right (4x1).
    # B is 5x5. Split into B_top (4x5) and B_bottom (1x5).
    # B_top splits into B_top_left (4x4) and B_top_right (4x1).
    # B_bottom splits into B_bottom_left (1x4) and B_bottom_right (1x1).
    
    # C = A * B = A_left * B_top + A_right * B_bottom
    # C = A_left * (B_top_left + B_top_right) + A_right * (B_bottom_left + B_bottom_right)
    # C = A_left * B_top_left + A_left * B_top_right + A_right * B_bottom_left + A_right * B_bottom_right
    
    # 1. A_left (4x4) * B_top_left (4x4) -> C_left (4x4). This is the Strassen part (Rank 49).
    # 2. A_left (4x4) * B_top_right (4x1) -> C_right_top (4x1). Schoolbook Rank 16.
    # 3. A_right (4x1) * B_bottom_left (1x4) -> C_right_bottom (4x1). Schoolbook Rank 16.
    # 4. A_right (4x1) * B_bottom_right (1x1) -> C_bottom_right (4x1). Schoolbook Rank 4.
    
    # Total Rank = 49 + 16 + 16 + 4 = 85.
    
    # Part 2: A_left (4x4) * B_top_right (4x1)
    # A[i, j] for i in 0..3, j in 0..3
    # B[j, 4] for j in 0..3
    # C[i, 4] for i in 0..3
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break
    
    # Part 3: A_right (4x1) * B_bottom_left (1x4)
    # A[i, 4] for i in 0..3
    # B[4, k] for k in 0..3
    # C[i, k] for i in 0..3, k in 0..3
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            if time.time() - start_time > deadline:
                break
            
    # Part 4: A_right (4x1) * B_bottom_right (1x1)
    # A[i, 4] for i in 0..3
    # B[4, 4]
    # C[i, 4] for i in 0..3
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
