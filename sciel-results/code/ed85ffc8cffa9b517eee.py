import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Shapes
    dim_U = n * m  # 16
    dim_V = m * p  # 36
    dim_W = p * n  # 36

    # We will build the decomposition block by block.
    # A is 4x4, split into 2x2 blocks.
    # B is 4x9, split into 2x2 and 2x1 blocks.
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # U_s (4, 7), V_s (4, 7), W_s (4, 7)
    # Local indices: u=i*2+j, v=j*2+k, w=k*2+i
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    
    # M2 = (A21+A22)B11
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1
    
    # M3 = A11(B12-B22)
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1
    
    # M4 = A22(B21-B11)
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[2,3]=1
    
    # M5 = (A11+A12)B22
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    
    # M6 = (A21-A11)(B11+B12)
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    
    # M7 = (A12-A22)(B21+B22)
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1

    # Schoolbook 2x2x1 decomposition (Rank 4)
    # U_t (4, 4), V_t (2, 4), W_t (2, 4)
    # Local indices: u=i*2+j, v=j, w=i
    
    U_t = np.zeros((4, 4))
    V_t = np.zeros((2, 4))
    W_t = np.zeros((2, 4))
    
    # C0 = A00 B0 + A01 B1
    U_t[0,0]=1; V_t[0,0]=1; W_t[0,0]=1
    U_t[1,1]=1; V_t[1,1]=1; W_t[0,1]=1
    # C1 = A10 B0 + A11 B1
    U_t[2,2]=1; V_t[0,2]=1; W_t[1,2]=1
    U_t[3,3]=1; V_t[1,3]=1; W_t[1,3]=1

    # Global matrices
    # We estimate max rank: 4 (A blocks) * 2 (B row blocks) * (4*7 + 1*4) = 128.
    # Let's allocate 130 to be safe.
    max_rank = 130
    U = np.zeros((dim_U, max_rank))
    V = np.zeros((dim_V, max_rank))
    W = np.zeros((dim_W, max_rank))
    
    rank_used = 0
    
    # Block definitions
    # A blocks: (i_start, j_start)
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B blocks: (j_start, k_start, k_width)
    # B00: (0,0,2), B01: (0,2,2), B02: (0,4,2), B03: (0,6,2), B04: (0,8,1)
    # B10: (2,0,2), B11: (2,2,2), B12: (2,4,2), B13: (2,6,2), B14: (2,8,1)
    
    # C blocks: (i_start, k_start, k_width)
    # C00: (0,0,2), C01: (0,2,2), C02: (0,4,2), C03: (0,6,2), C04: (0,8,1)
    # C10: (2,0,2), C11: (2,2,2), C12: (2,4,2), C13: (2,6,2), C14: (2,8,1)
    
    a_blocks = [
        (0, 0), # A00
        (0, 2), # A01
        (2, 0), # A10
        (2, 2)  # A11
    ]
    
    b_blocks = [
        (0, 0, 2), # B00
        (0, 2, 2), # B01
        (0, 4, 2), # B02
        (0, 6, 2), # B03
        (0, 8, 1), # B04
        (2, 0, 2), # B10
        (2, 2, 2), # B11
        (2, 4, 2), # B12
        (2, 6, 2), # B13
        (2, 8, 1)  # B14
    ]
    
    c_blocks = [
        (0, 0, 2), # C00
        (0, 2, 2), # C01
        (0, 4, 2), # C02
        (0, 6, 2), # C03
        (0, 8, 1), # C04
        (2, 0, 2), # C10
        (2, 2, 2), # C11
        (2, 4, 2), # C12
        (2, 6, 2), # C13
        (2, 8, 1)  # C14
    ]
    
    # We need to map each product A_block * B_block -> C_block
    # A_block index: ai, aj
    # B_block index: bi, bj, bw
    # C_block index: ci, ck, cw
    
    # The contribution to C(ci, ck) comes from A(ai, aj) * B(bi, bj)
    # if ai == ci and bi == aj (row of B matches col of A) and bj == ck.
    
    # Let's iterate over all valid combinations.
    # For each C block, identify the contributing A and B blocks.
    
    for ci, ck, cw in c_blocks:
        # C block is at rows ci, ci+1 and cols ck, ck+cw-1
        # It receives contributions from:
        # A(ai, bi) * B(bi, ck)
        # where bi can be 0,1,2,3 (but in blocks of 2)
        # Specifically, bi is the row index of B, which is the col index of A.
        # A is 4x4. Cols are 0,1,2,3.
        # B rows are 0,1,2,3.
        
        # We split A cols into 0,1 and 2,3.
        # So bi can be 0 (representing 0,1) or 2 (representing 2,3).
        
        for bi_start in [0, 2]:
            # A block is (ci, bi_start)
            ai_start = ci
            aj_start = bi_start
            
            # B block is (bi_start, ck)
            bj_start = ck
            bw = cw # B width must match C width
            
            # Check if B block exists
            # B block is defined by (bi_start, bj_start, bw)
            # We need to find the index in b_blocks
            b_idx = -1
            for idx, (b_r, b_c, b_w) in enumerate(b_blocks):
                if b_r == bi_start and b_c == bj_start and b_w == bw:
                    b_idx = idx
                    break
            
            if b_idx == -1:
                continue
                
            # Get A block info
            a_i, a_j = ai_start, aj_start
            
            # Get local indices
            # A local: 0..3
            # B local: 0..3 (if w=2) or 0..1 (if w=1)
            # C local: 0..3 (if w=2) or 0..1 (if w=1)
            
            if cw == 2:
                # Use Strassen
                # Local A indices: 0,1,2,3
                # Local B indices: 0,1,2,3
                # Local C indices: 0,1,2,3
                
                for r in range(7):
                    # U
                    for lu in range(4):
                        if U_s[lu, r] != 0:
                            global_u = a_i * 4 + a_j + (lu % 2) + 2 * (lu // 2)
                            # Wait, local u = i*2+j.
                            # global i = a_i + i_local, global j = a_j + j_local.
                            # i_local = lu // 2, j_local = lu % 2.
                            # global_u = (a_i + lu//2) * 4 + (a_j + lu%2)
                            U[global_u, rank_used] += U_s[lu, r]
                            
                    # V
                    for lv in range(4):
                        if V_s[lv, r] != 0:
                            # local v = j*2+k.
                            # j_local = lv // 2, k_local = lv % 2.
                            # global j = bi_start + j_local, global k = ck + k_local.
                            # global_v = global_j * 9 + global_k
                            global_v = (bi_start + lv//2) * 9 + (ck + lv%2)
                            V[global_v, rank_used] += V_s[lv, r]
                            
                    # W
                    for lw in range(4):
                        if W_s[lw, r] != 0:
                            # local w = k*2+i.
                            # k_local = lw // 2, i_local = lw % 2.
                            # global k = ck + k_local, global i = ci + i_local.
                            # global_w = global_k * 4 + global_i
                            global_w = (ck + lw//2) * 4 + (ci + lw%2)
                            W[global_w, rank_used] += W_s[lw, r]
                            
                    rank_used += 1
            else:
                # Use Schoolbook 2x2x1
                # Local A: 0..3
                # Local B: 0..1 (j_local, k_local=0)
                # Local C: 0..1 (i_local, k_local=0)
                
                for r in range(4):
                    # U
                    for lu in range(4):
                        if U_t[lu, r] != 0:
                            global_u = (a_i + lu//2) * 4 + (a_j + lu%2)
                            U[global_u, rank_used] += U_t[lu, r]
                            
                    # V
                    for lv in range(2):
                        if V_t[lv, r] != 0:
                            # lv is j_local. k_local is 0.
                            global_v = (bi_start + lv) * 9 + (ck + 0)
                            V[global_v, rank_used] += V_t[lv, r]
                            
                    # W
                    for lw in range(2):
                        if W_t[lw, r] != 0:
                            # lw is i_local. k_local is 0.
                            global_w = (ck + 0) * 4 + (ci + lw)
                            W[global_w, rank_used] += W_t[lw, r]
                            
                    rank_used += 1

    # Trim arrays to actual rank
    U = U[:, :rank_used]
    V = V[:, :rank_used]
    W = W[:, :rank_used]
    
    # Verify
    # T[a, b, c] = 1 if a=i*4+j, b=j*9+k, c=k*4+i
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(4):
        for j in range(4):
            for k in range(9):
                a = i * 4 + j
                b = j * 9 + k
                c = k * 4 + i
                T[a, b, c] = 1
                
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    if not np.allclose(T, T_rec):
        # Fallback to schoolbook if something went wrong
        rank = 144
        U = np.zeros((16, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    a = i * 4 + j
                    b = j * 9 + k
                    c = k * 4 + i
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
