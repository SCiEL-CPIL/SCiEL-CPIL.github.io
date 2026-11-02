"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen's algorithm for 2x2 block structure with 3x3 blocks.
    # 6x6 matrix = 2x2 blocks of 3x3 matrices.
    # Strassen uses 7 multiplications of 3x3 matrices.
    # Each 3x3 multiplication uses schoolbook rank 27.
    # Total rank = 7 * 27 = 189.
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 3  # 3x3 blocks
    num_blocks = 2  # 2x2 block structure
    
    # Schoolbook decomposition for 3x3 matrix multiplication
    # T_{3,3,3} has rank 27
    def schoolbook_3x3():
        bs = block_size
        rank_bs = bs * bs * bs  # 27
        U_bs = [[0] * rank_bs for _ in range(bs * bs)]
        V_bs = [[0] * rank_bs for _ in range(bs * bs)]
        W_bs = [[0] * rank_bs for _ in range(bs * bs)]
        col = 0
        for i in range(bs):
            for j in range(bs):
                for k in range(bs):
                    U_bs[i * bs + j][col] = 1
                    V_bs[j * bs + k][col] = 1
                    W_bs[k * bs + i][col] = 1
                    col += 1
        return U_bs, V_bs, W_bs, rank_bs
    
    U_bs, V_bs, W_bs, rank_bs = schoolbook_3x3()
    
    # Strassen's algorithm coefficients for 2x2 block matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication, we need to define:
    # - Which A blocks (with signs) go into U
    # - Which B blocks (with signs) go into V
    # - Which C blocks (with signs) go into W
    
    # Block indices: A11=0, A12=1, A21=2, A22=3
    # Same for B and C blocks.
    
    # Each Strassen term: list of (block_idx, sign) for A, B, C
    strassen_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # M1 = (A11+A22)(B11+B22) -> contributes to C11(+1), C22(+1)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A21+A22)B11 -> contributes to C21(+1), C22(-1)
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A11(B12-B22) -> contributes to C12(+1), C22(+1)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A22(B21-B11) -> contributes to C11(+1), C21(+1)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A11+A12)B22 -> contributes to C11(-1), C12(+1)
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A21-A11)(B11+B12) -> contributes to C22(+1)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A12-A22)(B21+B22) -> contributes to C11(+1)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    total_rank = 7 * rank_bs  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    # For each Strassen term t (0..6), we have rank_bs columns
    for t, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_terms):
        base_col = t * rank_bs
        
        # For each rank-1 component of the 3x3 schoolbook decomposition
        for r in range(rank_bs):
            col = base_col + r
            
            # U: sum over A blocks with signs
            for (block_idx, sign) in a_blocks:
                # block_idx: 0=A11, 1=A12, 2=A21, 3=A22
                # Row in U: block_row * 9 + local_row
                block_row = block_idx // 2  # 0 for A11,A12; 1 for A21,A22
                block_col = block_idx % 2   # 0 for A11,A21; 1 for A12,A22
                for local_row in range(block_size):
                    for local_col in range(block_size):
                        u_row = (block_row * block_size + local_row) * (block_size * 2) + (block_col * block_size + local_col)
                        # Wait, this indexing is wrong. Let me reconsider.
                        pass
            
            # Let me redo the indexing properly.
            # A is 6x6, viewed as 2x2 blocks of 3x3.
            # A[i,j] where i = block_row*3 + local_i, j = block_col*3 + local_j
            # block_row, block_col in {0,1}
            # U has rows indexed by (i, j) for the A matrix: i*m + j = i*6 + j
            # For a given A block (br, bc), local indices (li, lj):
            #   global i = br*3 + li, global j = bc*3 + lj
            #   U row = i*6 + j = (br*3 + li)*6 + (bc*3 + lj)
            
            # Similarly for V (B matrix): V row = j*6 + k where j = br*3+li, k = bc*3+lj
            # For B block (br, bc), local (li, lj): global j = br*3+li, global k = bc*3+lj
            #   V row = j*6 + k = (br*3+li)*6 + (bc*3+lj)
            
            # For W (C matrix): W row = k*6 + i where k = br*3+li, i = bc*3+lj
            # For C block (br, bc), local (li, lj): global k = br*3+li, global i = bc*3+lj
            #   W row = k*6 + i = (br*3+li)*6 + (bc*3+lj)
            
            # The 3x3 schoolbook decomposition: U_bs[li*3+lj, r], V_bs[lj*3+lk, r], W_bs[lk*3+li, r]
            # So for each local (li, lj, lk) with U_bs[li*3+lj, r] != 0, etc.
            
            # Actually, in the schoolbook, for each (li, lj, lk):
            # U_bs[li*3+lj, r] = 1, V_bs[lj*3+lk, r] = 1, W_bs[lk*3+li, r] = 1
            # where r = li*9 + lj*3 + lk
            
            for li in range(block_size):
                for lj in range(block_size):
                    for lk in range(block_size):
                        r_local = li * 9 + lj * 3 + lk
                        # U_bs[li*3+lj, r_local] = 1, V_bs[lj*3+lk, r_local] = 1, W_bs[lk*3+li, r_local] = 1
                        
                        # U contribution: for each A block in a_blocks
                        for (block_idx, sign) in a_blocks:
                            br = block_idx // 2
                            bc = block_idx % 2
                            global_i = br * block_size + li
                            global_j = bc * block_size + lj
                            u_row = global_i * 6 + global_j
                            U[u_row][col] += sign * 1  # U_bs value is 1
                        
                        # V contribution: for each B block in b_blocks
                        for (block_idx, sign) in b_blocks:
                            br = block_idx // 2
                            bc = block_idx % 2
                            global_j = br * block_size + lj
                            global_k = bc * block_size + lk
                            v_row = global_j * 6 + global_k
                            V[v_row][col] += sign * 1  # V_bs value is 1
                        
                        # W contribution: for each C block in c_blocks
                        for (block_idx, sign) in c_blocks:
                            br = block_idx // 2
                            bc = block_idx % 2
                            global_k = br * block_size + lk
                            global_i = bc * block_size + li
                            w_row = global_k * 6 + global_i
                            W[w_row][col] += sign * 1  # W_bs value is 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

