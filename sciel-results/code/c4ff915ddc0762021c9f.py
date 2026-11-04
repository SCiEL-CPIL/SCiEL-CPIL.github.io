"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 4x5x5
    # Split into 4x4x4 (rank 49), 4x4x1 (rank 1), 4x1x5 (rank 1)
    
    # Rank 1 term for M[4,4,1]
    # A is 4x4, B is 4x1. Result 4x1.
    # T[i, j, 0] = 1 for i in 0..3, j in 0..3.
    # U[i, col] = 1, V[j, col] = 1, W[0, col] = 1?
    # W index is k*n + i. Here k=0, i in 0..3.
    # So W[0*4+i] = 1.
    
    # Rank 1 term for M[4,1,5]
    # A is 4x1, B is 1x5. Result 4x5.
    # T[i, 0, k] = 1 for i in 0..3, k in 0..4.
    # U[i, col] = 1, V[0, col] = 1, W[k*4+i, col] = 1.
    
    # Rank 49 term for M[4,4,4]
    # Nested Strassen
    
    # Helper to add a rank-1 tensor
    def add_rank1(U, V, W, i, j, k, val=1):
        # i in 0..n-1, j in 0..m-1, k in 0..p-1
        # But we are building for specific sub-tensors.
        pass

    # We will explicitly construct the 49 terms for 4x4x4
    # and 2 terms for the rest.
    
    # Total rank = 49 + 1 + 1 = 51
    rank = 51
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # 1. M[4,4,4] part (indices i<4, j<4, k<4)
    # Nested Strassen
    # Outer Strassen on 2x2 blocks of 2x2 matrices
    # Inner Strassen on 2x2 matrices
    
    # Outer Strassen combinations (7)
    # Each combination is a product of two 2x2 matrices (which are sums of 2x2 blocks)
    # We need to expand each 2x2 matrix product into 7 rank-1 terms.
    
    # Let's define the 7 Strassen multipliers for 2x2 matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For the 4x4 case, A is [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The outer Strassen computes 7 products of 2x2 matrices.
    # Let P_s be the s-th outer product (s=0..6).
    # P_s = X_s * Y_s where X_s, Y_s are 2x2 matrices (sums of blocks).
    # We decompose X_s * Y_s using inner Strassen.
    
    # We need to track the coefficients.
    # This is complex to do symbolically in the loop.
    # Instead, we can pre-compute the 49 rank-1 tensors for 4x4x4.
    # Or, we can just use the 56-rank one (8 multiplications of 2x2 blocks, each rank 7)
    # 8 * 7 = 56.
    # 56 + 1 + 1 = 58.
    # This is easier to implement correctly.
    
    # Let's implement 58-rank decomposition.
    # 8 block multiplications:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # Each is a sum of two 2x2 matrix products.
    # Each 2x2 matrix product has rank 7.
    # So 8 products * 7 = 56 terms.
    
    # We need to map the 2x2 matrix product rank-1 terms to the 4x4 indices.
    
    # Let's define the 7 rank-1 tensors for a 2x2 multiplication.
    # Indices i, j, k in {0, 1}.
    # T[i, j, k]
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Result C:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # We need to express each M_s as a sum of rank-1 tensors.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This is a sum of 4 scalar multiplications.
    # But we want rank-1 tensors for the 2x2 matrix multiplication.
    # The tensor for 2x2 multiplication is T2.
    # T2 = M1 + M2 + ...? No.
    # The decomposition is:
    # T2 = u1 v1 w1 + ... + u7 v7 w7
    # where the sum equals the 2x2 multiplication tensor.
    
    # The 7 rank-1 tensors for 2x2 multiplication are:
    # 1. (1,1,1) coeff 1? No.
    # Let's list the 7 vectors (u, v, w) for 2x2 Strassen.
    # u, v, w are in R^4 (flattened 2x2).
    
    # M1: (A00+A11)(B00+B11)
    # u1 = [1, 0, 0, 1] (A00+A11)
    # v1 = [1, 0, 0, 1] (B00+B11)
    # w1 = [1, 0, 0, 1]? No, w is the output coefficients.
    # The contribution to C is M1 * (1, 0, 0, 1) in C00, C11?
    # C00 += M1, C11 += M1.
    # So w1 = [1, 0, 0, 1] (C00, C01, C10, C11).
    
    # M2: (A10+A11)B00
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # w2 = [0, 0, 1, 0] (C10 += M2)
    
    # M3: A00(B10-B11)
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 0, 1, -1]
    # w3 = [0, 1, 0, 0] (C01 += M3)
    
    # M4: A11(B01-B00)
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 1, -1, 0]
    # w4 = [-1, 0, 1, 0] (C00 -= M4, C10 += M4)
    
    # M5: (A00+A01)B11
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # w5 = [0, 1, 0, 1] (C01 += M5, C11 += M5)
    
    # M6: (A10-A00)(B00+B01)
    # u6 = [-1, 0, 1, 0]
    # v6 = [1, 1, 0, 0]
    # w6 = [1, 0, 0, 0] (C00 += M6)
    
    # M7: (A01-A11)(B10+B11)
    # u7 = [0, 1, 0, -1]
    # v7 = [0, 0, 1, 1]
    # w7 = [-1, 0, 0, 1] (C00 -= M7, C11 += M7)
    
    # These 7 vectors form the rank-7 decomposition of 2x2 multiplication.
    
    # Now for 4x4 multiplication using 8 block products:
    # We have 8 products of 2x2 matrices.
    # Each product is decomposed into 7 rank-1 tensors.
    # Total 56.
    
    # Block indices:
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    
    # Products:
    # P0 = A00 B00 -> contributes to C00
    # P1 = A01 B10 -> contributes to C00
    # P2 = A00 B01 -> contributes to C01
    # P3 = A01 B11 -> contributes to C01
    # P4 = A10 B00 -> contributes to C10
    # P5 = A11 B10 -> contributes to C10
    # P6 = A10 B01 -> contributes to C11
    # P7 = A11 B11 -> contributes to C11
    
    # For each product P_idx, we add 7 rank-1 tensors.
    # The u, v vectors are scaled/shifted to the 4x4 indices.
    # The w vector is scaled/shifted to the 4x4 indices.
    
    # u, v are in R^4 (for 2x2).
    # In 4x4, the block A_ab occupies indices:
    # i = 2*a + i_l, j = 2*b + j_l.
    
    # We will iterate through the 8 products and the 7 Strassen terms.
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [-1, 0, 0, 1]
    ]
    
    # Products: (a_block, b_block, c_block)
    # a_block: (row_block, col_block) of A
    # b_block: (row_block, col_block) of B
    # c_block: (row_block, col_block) of C
    products = [
        ((0,0), (0,0), (0,0)), # P0
        ((0,1), (1,0), (0,0)), # P1
        ((0,0), (0,1), (0,1)), # P2
        ((0,1), (1,1), (0,1)), # P3
        ((1,0), (0,0), (1,0)), # P4
        ((1,1), (1,0), (1,0)), # P5
        ((1,0), (0,1), (1,1)), # P6
        ((1,1), (1,1), (1,1)), # P7
    ]
    
    for a_b, b_b, c_b in products:
        a_r, a_c = a_b
        b_r, b_c = b_b
        c_r, c_c = c_b
        
        for s in range(7):
            u_2x2 = strassen_u[s]
            v_2x2 = strassen_v[s]
            w_2x2 = strassen_w[s]
            
            # Expand u_2x2 to u_4x4 (size 16)
            u_4x4 = np.zeros(16)
            for il in range(2):
                for jl in range(2):
                    val = u_2x2[il*2+jl] # Wait, u_2x2 is flattened 2x2.
                    # u_2x2[0]=A00, u_2x2[1]=A01, u_2x2[2]=A10, u_2x2[3]=A11
                    # My strassen_u lists are [A00, A01, A10, A11]?
                    # Let's check strassen_u[0] = [1, 0, 0, 1].
                    # M1 = (A00+A11)(B00+B11).
                    # So u[0]=1 (A00), u[3]=1 (A11).
                    # So index 0 is A00, 1 is A01, 2 is A10, 3 is A11.
                    # In 4x4, A00 is at i=0,1; j=0,1.
                    # So for each element in u_2x2, we add val to the corresponding 2x2 block in u_4x4.
                    
            # Let's do this properly.
            u_4x4 = np.zeros(16)
            for idx_2x2 in range(4):
                val = u_2x2[idx_2x2]
                if val != 0:
                    r = idx_2x2 // 2
                    c = idx_2x2 % 2
                    # Block (r, c) in 4x4
                    for il in range(2):
                        for jl in range(2):
                            i = 2*r + il
                            j = 2*c + jl
                            u_4x4[i*4+j] += val
                            
            v_4x4 = np.zeros(16)
            for idx_2x2 in range(4):
                val = v_2x2[idx_2x2]
                if val != 0:
                    r = idx_2x2 // 2
                    c = idx_2x2 % 2
                    for il in range(2):
                        for jl in range(2):
                            i = 2*r + il
                            j = 2*c + jl
                            v_4x4[i*4+j] += val
                            
            w_4x4 = np.zeros(16)
            for idx_2x2 in range(4):
                val = w_2x2[idx_2x2]
                if val != 0:
                    r = idx_2x2 // 2
                    c = idx_2x2 % 2
                    for il in range(2):
                        for jl in range(2):
                            i = 2*r + il
                            j = 2*c + jl
                            # W index is k*n + i. Here n=4 for the sub-problem?
                            # No, W is for the full 4x5x5.
                            # For the 4x4x4 part, k is 0..3, i is 0..3.
                            # W index in full tensor: k*4 + i.
                            # But we are adding to the full W matrix of size 5*4=20.
                            # So W[4*k + i].
                            w_4x4[j*4+i] += val # Wait, w_2x2 is flattened C.
                            # C is 4x4.
                            # w_2x2[idx] corresponds to C[r,c] element?
                            # My strassen_w lists are [C00, C01, C10, C11].
                            # So idx 0 is C00, 1 is C01, 2 is C10, 3 is C11.
                            # So r = idx // 2, c = idx % 2.
                            # i = 2*r + il, j = 2*c + jl.
                            # W index = j*4 + i.
                            
            # Add to U, V, W
            U[:, col] = u_4x4
            V[:, col] = v_4x4
            W[:, col] = w_4x4
            col += 1
            
    # Add M[4,4,1] rank 1
    # A is 4x4, B is 4x1.
    # U: 1 for i=0..3, j=0..3.
    # V: 1 for j=0..3, k=0 (B is 4x1, so k=0).
    # W: 1 for k=0, i=0..3.
    U[0:16, col] = 1
    V[0:4, col] = 1 # B indices 0..3 (j=0..3, k=0)
    W[0:4, col] = 1 # W indices 0..3 (k=0, i=0..3)
    col += 1
    
    # Add M[4,1,5] rank 1
    # A is 4x1, B is 1x5.
    # U: 1 for i=0..3, j=0 (A is 4x1, so j=0).
    # V: 1 for j=0, k=0..4 (B is 1x5, so j=0, k=0..4).
    # W: 1 for k=0..4, i=0..3.
    U[0:4, col] = 1 # i=0..3, j=0
    V[0:5, col] = 1 # j=0, k=0..4
    for k in range(5):
        for i in range(4):
            W[k*4+i, col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
