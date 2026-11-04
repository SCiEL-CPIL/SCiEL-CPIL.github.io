import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    start_time = time.time()
    deadline = 120.0
    
    # Local Strassen tensors for 2x2x2
    # u, v, w are lists of 4 elements (indices 0..3 for 2x2 blocks)
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_222 = [
        { 'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1] }, # M1
        { 'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1] }, # M2
        { 'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1] }, # M3
        { 'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0] }, # M4
        { 'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0] }, # M5
        { 'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1] }, # M6
        { 'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0] }  # M7
    ]
    
    # Define the 7 Strassen products for 4x4x4
    # Each product is defined by:
    # - which A blocks are involved (and with what signs, though Strassen sums are linear so we just map the local u)
    # - which B blocks are involved
    # - which C blocks are involved (and with what signs)
    
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += M1, C11 += M1
    prod_444 = [
        { 'a_blocks': [(0,0,1), (1,1,1)], 'b_blocks': [(0,0,1), (1,1,1)], 'c_blocks': [(0,0,1), (1,1,1)] },
        { 'a_blocks': [(1,0,1), (1,1,1)], 'b_blocks': [(0,0,1)], 'c_blocks': [(1,0,1), (1,1,-1)] },
        { 'a_blocks': [(0,0,1)], 'b_blocks': [(0,1,1), (1,1,-1)], 'c_blocks': [(0,1,1), (1,1,1)] },
        { 'a_blocks': [(1,1,1)], 'b_blocks': [(0,0,-1), (1,0,1)], 'c_blocks': [(0,0,1), (1,0,1)] },
        { 'a_blocks': [(0,0,1), (0,1,1)], 'b_blocks': [(1,1,1)], 'c_blocks': [(0,0,-1), (0,1,1)] },
        { 'a_blocks': [(1,0,-1), (0,0,1)], 'b_blocks': [(0,0,1), (0,1,1)], 'c_blocks': [(1,1,1)] },
        { 'a_blocks': [(0,1,1), (1,1,-1)], 'b_blocks': [(1,0,1), (1,1,1)], 'c_blocks': [(0,0,1)] }
    ]
    
    # Define the 4 products for 4x4x2 (columns 4,5)
    # B is 4x2. Blocks B0 (rows 0,1), B1 (rows 2,3).
    # P1 = A00 * B0
    # P2 = A01 * B1
    # P3 = A10 * B0
    # P4 = A11 * B1
    # C0 += P1, C1 += P2 (for row block 0)
    # C2 += P3, C3 += P4 (for row block 1)
    # Note: C indices for 4x4x2 are k=4,5.
    # C block (0, 45) corresponds to i in 0,1 and k in 4,5.
    # C block (1, 45) corresponds to i in 2,3 and k in 4,5.
    
    prod_442 = [
        { 'a_blocks': [(0,0,1)], 'b_blocks': [(0,4,1)], 'c_blocks': [(0,4,1)] },
        { 'a_blocks': [(0,1,1)], 'b_blocks': [(1,4,1)], 'c_blocks': [(1,4,1)] },
        { 'a_blocks': [(1,0,1)], 'b_blocks': [(0,4,1)], 'c_blocks': [(2,4,1)] },
        { 'a_blocks': [(1,1,1)], 'b_blocks': [(1,4,1)], 'c_blocks': [(3,4,1)] }
    ]
    
    rank = 49 + 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def expand_tensor(local_u, local_v, local_w, a_blocks, b_blocks, c_blocks, col_idx):
        # local_u, v, w are lists of 4
        # a_blocks is list of (i_off, j_off, sign)
        # b_blocks is list of (j_off, k_off, sign)
        # c_blocks is list of (i_off, k_off, sign)
        
        # Accumulate into global vectors
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        # Map local indices to global
        # Local indices 0..3 correspond to (0,0), (0,1), (1,0), (1,1)
        # local_i = l // 2, local_j = l % 2 for u? No.
        # u index l: i_loc = l // 2, j_loc = l % 2.
        # v index l: j_loc = l // 2, k_loc = l % 2.
        # w index l: k_loc = l // 2, i_loc = l % 2.
        
        for l in range(4):
            val_u = local_u[l]
            val_v = local_v[l]
            val_w = local_w[l]
            
            i_loc = l // 2
            j_loc = l % 2
            k_loc = l // 2 # for v
            i_loc_w = l % 2 # for w
            k_loc_w = l // 2 # for w
            
            # Wait, v index l: j_loc = l // 2, k_loc = l % 2.
            # w index l: k_loc = l // 2, i_loc = l % 2.
            
            # For u:
            i_u_loc = l // 2
            j_u_loc = l % 2
            
            # For v:
            j_v_loc = l // 2
            k_v_loc = l % 2
            
            # For w:
            k_w_loc = l // 2
            i_w_loc = l % 2
            
            # Add to u_vec
            for (i_off, j_off, sign) in a_blocks:
                if sign == 0: continue
                i_glob = i_off + i_u_loc
                j_glob = j_off + j_u_loc
                idx_u = i_glob * m + j_glob
                u_vec[idx_u] += val_u * sign
                
            # Add to v_vec
            for (j_off, k_off, sign) in b_blocks:
                if sign == 0: continue
                j_glob = j_off + j_v_loc
                k_glob = k_off + k_v_loc
                idx_v = j_glob * p + k_glob
                v_vec[idx_v] += val_v * sign
                
            # Add to w_vec
            for (i_off, k_off, sign) in c_blocks:
                if sign == 0: continue
                i_glob = i_off + i_w_loc
                k_glob = k_off + k_w_loc
                idx_w = k_glob * n + i_glob
                w_vec[idx_w] += val_w * sign
        
        # Store in U, V, W
        for r in range(n * m):
            U[r][col_idx] = u_vec[r]
        for r in range(m * p):
            V[r][col_idx] = v_vec[r]
        for r in range(p * n):
            W[r][col_idx] = w_vec[r]

    # Process 4x4x4 part
    for prod in prod_444:
        for l in range(7):
            if time.time() > deadline - 5:
                break
            local = strassen_222[l]
            expand_tensor(local['u'], local['v'], local['w'], 
                          prod['a_blocks'], prod['b_blocks'], prod['c_blocks'], col)
            col += 1
            
    # Process 4x4x2 part
    for prod in prod_442:
        for l in range(7):
            if time.time() > deadline - 5:
                break
            local = strassen_222[l]
            expand_tensor(local['u'], local['v'], local['w'], 
                          prod['a_blocks'], prod['b_blocks'], prod['c_blocks'], col)
            col += 1
            
    return {"U": U, "V": V, "W": W}
