import numpy as np
import time

def construct():
    """
    Constructs a rank-77 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses a recursive Strassen-like approach:
    4x4*4x6 -> 7 terms of 2x2*2x3.
    2x2*2x3 -> 4 (schoolbook 2x2*2x1) + 7 (Strassen 2x2*2x2) = 11.
    Total Rank = 77.
    """
    # Dimensions
    n, m, p = 4, 4, 6
    
    # We will build the U, V, W matrices as numpy arrays
    # U: (n*m, rank) -> (16, 77)
    # V: (m*p, rank) -> (24, 77)
    # W: (p*n, rank) -> (24, 77)
    
    rank = 77
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    # Helper to create a rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1
    # U_sub: (4, 4), V_sub: (2, 4), W_sub: (2, 4)
    # Local indices:
    # A: 2x2, B: 2x1, C: 2x1
    # U index: i*2+j
    # V index: j (since p=1)
    # W index: k*2+i? No, W is for C. C is 2x1.
    # W index: k*1 + i? No, the convention is W[k*n+i].
    # For 2x2*2x1, n_local=2, m_local=2, p_local=1.
    # W shape (p_local * n_local, rank) = (2, rank).
    # Index k (0) * 2 + i (0..1).
    
    def build_2x2x1():
        u = np.zeros((4, 4), dtype=int)
        v = np.zeros((2, 4), dtype=int)
        w = np.zeros((2, 4), dtype=int)
        
        # Terms:
        # 0: A[0,0] * B[0] -> C[0]
        u[0, 0] = 1; v[0, 0] = 1; w[0, 0] = 1
        # 1: A[0,1] * B[1] -> C[0]
        u[1, 1] = 1; v[1, 1] = 1; w[0, 1] = 1
        # 2: A[1,0] * B[0] -> C[1]
        u[2, 2] = 1; v[0, 2] = 1; w[1, 2] = 1
        # 3: A[1,1] * B[1] -> C[1]
        u[3, 3] = 1; v[1, 3] = 1; w[1, 3] = 1
        return u, v, w

    # Helper to create a rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2
    def build_2x2x2():
        u = np.zeros((4, 7), dtype=int)
        v = np.zeros((4, 7), dtype=int)
        w = np.zeros((4, 7), dtype=int)
        
        # Standard Strassen
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
        
        # u indices: i*2+j
        # v indices: j*2+k
        # w indices: k*2+i
        
        # Term 0: M1
        u[0,0]=1; u[3,0]=1
        v[0,0]=1; v[3,0]=1
        w[0,0]=1; w[3,0]=1 # C[0,0] (k=0,i=0), C[1,1] (k=1,i=1) -> W[0], W[3]
        
        # Term 1: M2
        u[2,1]=1; u[3,1]=1
        v[0,1]=1
        w[2,1]=1; w[3,1]=-1 # C[1,0] (k=0,i=1) -> W[2], C[1,1] (k=1,i=1) -> W[3]
        
        # Term 2: M3
        u[0,2]=1
        v[1,2]=1; v[3,2]=-1
        w[1,2]=1; w[3,2]=1 # C[0,1] (k=1,i=0) -> W[1], C[1,1] (k=1,i=1) -> W[3]
        
        # Term 3: M4
        u[3,3]=1
        v[2,3]=1; v[0,3]=-1
        w[0,3]=1; w[2,3]=1 # C[0,0] (k=0,i=0) -> W[0], C[1,0] (k=0,i=1) -> W[2]
        
        # Term 4: M5
        u[0,4]=1; u[1,4]=1
        v[3,4]=1
        w[0,4]=-1; w[1,4]=1 # C[0,0] -> -1, C[0,1] -> 1
        
        # Term 5: M6
        u[2,5]=1; u[0,5]=-1
        v[0,5]=1; v[1,5]=1
        w[3,5]=1 # C[1,1]
        
        # Term 6: M7
        u[1,6]=1; u[3,6]=-1
        v[2,6]=1; v[3,6]=1
        w[0,6]=1 # C[0,0]

        return u, v, w

    # Combine to build 2x2 * 2x3 (Rank 11)
    def build_2x2x3():
        # U: (4, 11), V: (6, 11), W: (6, 11)
        # V indices: j*3+k (j in 0..1, k in 0..2)
        # W indices: k*2+i (k in 0..2, i in 0..1)
        
        u_11, v_11, w_11 = build_2x2x1()
        u_22, v_22, w_22 = build_2x2x2()
        
        u_total = np.zeros((4, 11), dtype=int)
        v_total = np.zeros((6, 11), dtype=int)
        w_total = np.zeros((6, 11), dtype=int)
        
        # Part 1: 2x2 * 2x1 (cols 0 of B, col 0 of C)
        u_total[:, 0:4] = u_11
        
        # Map V: V_11 (2x1 indices j) to V_total (2x3 indices j*3+k).
        # j=0 -> 0, j=1 -> 3.
        v_total[0, 0:4] = v_11[0]
        v_total[3, 0:4] = v_11[1]
        
        # Map W: W_11 (2x1 indices k*1+i? No, W_11 is for 2x1 result.
        # Result C is 2x1. W index in global 2x3 (p=2? No, p=2 for result dim? No, result is 2x3?
        # Wait, 2x2 * 2x1 -> 2x1 result.
        # In the global 2x2 * 2x3, the first column of C is 2x1.
        # W_global index for C[k, i] where k=0 (only col), i=0,1.
        # Formula: k*2 + i.
        # k=0, i=0 -> 0. k=0, i=1 -> 1.
        # W_11 rows correspond to C_0 and C_1.
        w_total[0, 0:4] = w_11[0]
        w_total[1, 0:4] = w_11[1]
        
        # Part 2: 2x2 * 2x2 (cols 1,2 of B, cols 1,2 of C)
        u_total[:, 4:11] = u_22
        
        # Map V: V_22 (2x2 indices j*2+k) to V_total (2x3 indices j*3+k+1).
        # j=0, k=0 -> V_22[0] -> V_total[0*3+1] = 1
        # j=0, k=1 -> V_22[1] -> V_total[0*3+2] = 2
        # j=1, k=0 -> V_22[2] -> V_total[1*3+1] = 4
        # j=1, k=1 -> V_22[3] -> V_total[1*3+2] = 5
        v_total[1, 4:11] = v_22[0]
        v_total[2, 4:11] = v_22[1]
        v_total[4, 4:11] = v_22[2]
        v_total[5, 4:11] = v_22[3]
        
        # Map W: W_22 (2x2 indices k*2+i) to W_total (2x3 indices k*2+i).
        # Wait, W_total is for 2x3 result (p=3? No, result is 2x3?
        # Input B is 2x3. Result C is 2x3.
        # W index: k (col of C) * 2 (rows) + i (row).
        # W_22 is for 2x2 result (cols 1,2 of global C).
        # Local k=0 (col 1 of global), i=0,1 -> W_global[0*2+0]=0? No.
        # Global C col 0 is from Part 1.
        # Global C col 1 is from Part 2 local col 0.
        # Global C col 2 is from Part 2 local col 1.
        # Local W_22 indices: k_loc (0,1) * 2 + i.
        # k_loc=0, i=0 -> 0. Maps to Global Col 1, Row 0.
        # Global Index: (1) * 2 + 0 = 2.
        # k_loc=0, i=1 -> 1. Maps to Global Col 1, Row 1.
        # Global Index: (1) * 2 + 1 = 3.
        # k_loc=1, i=0 -> 2. Maps to Global Col 2, Row 0.
        # Global Index: (2) * 2 + 0 = 4.
        # k_loc=1, i=1 -> 3. Maps to Global Col 2, Row 1.
        # Global Index: (2) * 2 + 1 = 5.
        
        w_total[2, 4:11] = w_22[0]
        w_total[3, 4:11] = w_22[1]
        w_total[4, 4:11] = w_22[2]
        w_total[5, 4:11] = w_22[3]
        
        return u_total, v_total, w_total

    u_sub, v_sub, w_sub = build_2x2x3()
    
    # Now build 4x4 * 4x6
    # 7 Strassen terms.
    # Each term uses u_sub, v_sub, w_sub.
    
    # Offsets for A (4x4) blocks (2x2)
    # 0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)
    a_off = [(0,0), (0,2), (2,0), (2,2)]
    # Offsets for B (4x6) blocks (2x3)
    # 0: (0,0), 1: (0,3), 2: (2,0), 3: (2,3)
    b_off = [(0,0), (0,3), (2,0), (2,3)]
    # Offsets for C (4x6) blocks (2x3)
    # W is p x n = 6 x 4.
    # Block 0: (0,0), 1: (0,2), 2: (3,0), 3: (3,2) ??
    # C is 4x6.
    # C[0:2, 0:3] -> Block 0.
    # C[0:2, 3:6] -> Block 1.
    # C[2:4, 0:3] -> Block 2.
    # C[2:4, 3:6] -> Block 3.
    # W index: k*n + i.
    # Block 0: k in 0..2, i in 0..1.
    # W rows for Block 0: 0..5.
    # Block 1: k in 0..2, i in 0..1.
    # Wait, W is indexed by output col k and output row i?
    # Standard: W[k*n + i].
    # C is n x p. Output row i, col k.
    # So W row index = k * n + i.
    # Block 0 (C rows 0:2, cols 0:3): i in 0,1. k in 0,1,2.
    # Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5, 2*4+0=8, 2*4+1=9.
    # This is not contiguous!
    # My sub-tensor W_sub is for 2x3 result (n=2, p=3? No, n=2 rows, p=3 cols?
    # In build_2x2x3, n_sub=2, m_sub=2, p_sub=3.
    # W_sub shape (p_sub * n_sub, rank) = (6, 11).
    # Index k * n_sub + i.
    # k in 0..2, i in 0..1.
    # Indices: 0, 1, 2, 3, 4, 5?
    # k=0: 0, 1. k=1: 2, 3. k=2: 4, 5.
    # This matches local block mapping.
    
    # When mapping to global 4x4x6:
    # Global n=4, p=6.
    # Block 0 (top-left of C): i_global 0..1, k_global 0..2.
    # W_global index: k_global * 4 + i_global.
    # Local i 0->0, 1->1. Local k 0->0, 1->1, 2->2.
    # Mapping:
    # L0 (k=0,i=0) -> G(0*4+0)=0
    # L1 (k=0,i=1) -> G(0*4+1)=1
    # L2 (k=1,i=0) -> G(1*4+0)=4
    # L3 (k=1,i=1) -> G(1*4+1)=5
    # L4 (k=2,i=0) -> G(2*4+0)=8
    # L5 (k=2,i=1) -> G(2*4+1)=9
    
    # Block 1 (top-right of C): i_global 0..1, k_global 3..5.
    # L0 -> G(3*4+0)=12
    # L1 -> G(3*4+1)=13
    # L2 -> G(4*4+0)=16
    # L3 -> G(4*4+1)=17
    # L4 -> G(5*4+0)=20
    # L5 -> G(5*4+1)=21
    
    # Block 2 (bottom-left): i_global 2..3, k_global 0..2.
    # L0 -> G(0*4+2)=2
    # L1 -> G(0*4+3)=3
    # L2 -> G(1*4+2)=6
    # L3 -> G(1*4+3)=7
    # L4 -> G(2*4+2)=10
    # L5 -> G(2*4+3)=11
    
    # Block 3 (bottom-right): i_global 2..3, k_global 3..5.
    # L0 -> G(3*4+2)=14
    # L1 -> G(3*4+3)=15
    # L2 -> G(4*4+2)=18
    # L3 -> G(4*4+3)=19
    # L4 -> G(5*4+2)=22
    # L5 -> G(5*4+3)=23

    # Define mapping for W
    w_maps = [
        [0, 1, 4, 5, 8, 9],   # Block 0
        [12, 13, 16, 17, 20, 21], # Block 1
        [2, 3, 6, 7, 10, 11], # Block 2
        [14, 15, 18, 19, 22, 23]  # Block 3
    ]
    
    # Define Strassen Terms
    terms = [
        # A blocks, B blocks, C blocks
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]), # M1
        ([ (2,1), (3,1) ], [ (0,1) ],       [ (2,1), (3,-1) ]), # M2
        ([ (0,1) ],         [ (1,1), (3,-1) ], [ (1,1), (3,1) ]), # M3
        ([ (3,1) ],         [ (2,1), (0,-1) ], [ (0,1), (2,1) ]), # M4
        ([ (0,1), (1,1) ], [ (3,1) ],       [ (0,-1), (1,1) ]), # M5
        ([ (2,1), (0,-1) ],[ (0,1), (1,1) ], [ (3,1) ]),         # M6
        ([ (1,1), (3,-1) ],[ (2,1), (3,1) ], [ (0,1) ]),         # M7
    ]
    
    for t_idx, (A_b, B_b, C_b) in enumerate(terms):
        base_col = t_idx * 11
        
        # Update U
        for (blk_a, coeff_a) in A_b:
            ro_a, co_a = a_off[blk_a]
            for i in range(4):
                for j in range(4):
                    # i is global row, j is global col
                    # Check if i,j falls in block blk_a
                    # i in [ro_a, ro_a+1], j in [co_a, co_a+1]
                    if ro_a <= i < ro_a+2 and co_a <= j < co_a+2:
                        i_loc = i - ro_a
                        j_loc = j - co_a
                        u_val = u_sub[i_loc*2 + j_loc, :]
                        if np.any(u_val != 0):
                            U[i*4+j, base_col:base_col+11] += coeff_a * u_val
        
        # Update V
        for (blk_b, coeff_b) in B_b:
            ro_b, co_b = b_off[blk_b]
            for j in range(4):
                for k in range(6):
                    if ro_b <= j < ro_b+2 and co_b <= k < co_b+3:
                        j_loc = j - ro_b
                        k_loc = k - co_b
                        v_val = v_sub[j_loc*3 + k_loc, :]
                        if np.any(v_val != 0):
                            V[j*6+k, base_col:base_col+11] += coeff_b * v_val
        
        # Update W
        for (blk_c, coeff_c) in C_b:
            # blk_c determines which w_map to use
            w_map = w_maps[blk_c]
            for i in range(4):
                for k in range(6):
                    # W row index in sub tensor corresponds to local i, k
                    # We need to find if this global (i,k) is in the block
                    # Block 0: i in 0..1, k in 0..2
                    # Block 1: i in 0..1, k in 3..5
                    # Block 2: i in 2..3, k in 0..2
                    # Block 3: i in 2..3, k in 3..5
                    
                    if blk_c == 0 and i < 2 and k < 3:
                        i_loc = i
                        k_loc = k
                    elif blk_c == 1 and i < 2 and k >= 3:
                        i_loc = i
                        k_loc = k - 3
                    elif blk_c == 2 and i >= 2 and k < 3:
                        i_loc = i - 2
                        k_loc = k
                    elif blk_c == 3 and i >= 2 and k >= 3:
                        i_loc = i - 2
                        k_loc = k - 3
                    else:
                        continue
                    
                    w_row_sub = k_loc * 2 + i_loc
                    w_val = w_sub[w_row_sub, :]
                    if np.any(w_val != 0):
                        # Map to global W row
                        # w_map is list of global rows for sub rows 0..5
                        # sub_row 0 corresponds to L0 (k=0,i=0)
                        # So w_row_sub is the index in w_map?
                        # Yes, w_row_sub is 0..5.
                        g_w_row = w_map[w_row_sub]
                        W[g_w_row, base_col:base_col+11] += coeff_c * w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
