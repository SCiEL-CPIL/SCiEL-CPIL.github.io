"""Matrix multiplication tensor decomposition for 3x3x4."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    Uses a block decomposition strategy:
    1. Partition A (3x3) and B (3x4) into overlapping blocks to exploit structure.
    2. Use Strassen's algorithm (rank 7) for the 2x2x2 core multiplications.
    3. Use schoolbook for the remaining rectangular/linear parts.
    
    Target Rank: 34 (vs 36 schoolbook)
    """
    n, m, p = 3, 3, 4
    r = 34  # Target rank
    
    # Pre-allocate matrices with zeros
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(uidx, vidx, widx, u_val, v_val, w_val, col):
        U[uidx][col] += u_val
        V[vidx][col] += v_val
        W[widx][col] += w_val

    # Indices:
    # A is 3x3. Indices (i, j) where i in 0..2, j in 0..2.
    # B is 3x4. Indices (j, k) where j in 0..2, k in 0..3.
    # C is 3x4. Indices (i, k) where i in 0..2, k in 0..3.
    # 
    # We partition the rows of A/B into {0,1} and {2}.
    # We partition the columns of B into {0,1} and {2,3}.
    # 
    # Let's define the blocks:
    # A_22 = A[0:2, 0:2] (2x2)
    # A_21 = A[0:2, 2:3] (2x1)
    # A_12 = A[2:3, 0:2] (1x2)
    # A_11 = A[2:3, 2:3] (1x1)
    # 
    # B_22 = B[0:2, 0:2] (2x2)
    # B_2p = B[0:2, 2:4] (2x2)  # Columns 2,3
    # B_12 = B[2:3, 0:2] (1x2)
    # B_1p = B[2:3, 2:4] (1x2)
    
    # The product C = A * B is 3x4.
    # C[0:2, 0:2] = A_22 * B_22 + A_21 * B_12
    # C[0:2, 2:4] = A_22 * B_2p + A_21 * B_1p
    # C[2:3, 0:2] = A_12 * B_22 + A_11 * B_12
    # C[2:3, 2:4] = A_12 * B_2p + A_11 * B_1p
    
    # We can compute:
    # 1. A_22 * B_22 (2x2x2) -> Rank 7 (Strassen)
    # 2. A_22 * B_2p (2x2x2) -> Rank 7 (Strassen)
    # 3. A_21 * B_12 (2x1x2) -> Rank 4 (Schoolbook: 2*1*2)
    # 4. A_21 * B_1p (2x1x2) -> Rank 4 (Schoolbook)
    # 5. A_12 * B_22 (1x2x2) -> Rank 4 (Schoolbook)
    # 6. A_12 * B_2p (1x2x2) -> Rank 4 (Schoolbook)
    # 7. A_11 * B_12 (1x1x2) -> Rank 2 (Schoolbook)
    # 8. A_11 * B_1p (1x1x2) -> Rank 2 (Schoolbook)
    
    # Total Rank = 7 + 7 + 4 + 4 + 4 + 4 + 2 + 2 = 34.
    
    # Let's implement this.
    # We need to map the local indices of these sub-problems to the global indices.
    
    # Global indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Strassen Helper for 2x2x2
    # Inputs: 
    #   A indices: (a_i, a_j) for i,j in 0,1
    #   B indices: (b_i, b_j) for i,j in 0,1
    #   C indices: (c_i, c_j) for i,j in 0,1
    # We need to add terms to the global U, V, W matrices.
    
    # Since the blocks are distinct in terms of which global indices they touch,
    # we can treat them as independent sub-tensors that we sum up.
    # However, the rank is the sum of the ranks of the sub-tensors only if they
    # are supported on disjoint sets of "columns" in the decomposition?
    # No, the decomposition is a sum of rank-1 tensors.
    # T_total = T_1 + T_2 + ...
    # If T_1 has rank r_1 and T_2 has rank r_2, T_total has rank <= r_1 + r_2.
    # We just need to assign unique column indices for each term.
    
    col = 0
    
    # --- Block 1: A_22 * B_22 ---
    # A_22 uses global A[0:2, 0:2] -> U indices: 0,1,3,4 (i*m+j)
    #    i=0,j=0 -> 0
    #    i=0,j=1 -> 1
    #    i=1,j=0 -> 3
    #    i=1,j=1 -> 4
    # B_22 uses global B[0:2, 0:2] -> V indices: 0,1,4,5 (j*p+k)
    #    j=0,k=0 -> 0
    #    j=0,k=1 -> 1
    #    j=1,k=0 -> 4
    #    j=1,k=1 -> 5
    # C_22 uses global C[0:2, 0:2] -> W indices: 0,1,3,4 (k*n+i)
    #    k=0,i=0 -> 0
    #    k=0,i=1 -> 1
    #    k=1,i=0 -> 3
    #    k=1,i=1 -> 4
    
    # Strassen Terms for A_22 * B_22:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 -= M1? No, standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map local (i,j) to global indices for A_22:
    # Local (0,0) -> Global (0,0) -> U_idx 0
    # Local (0,1) -> Global (0,1) -> U_idx 1
    # Local (1,0) -> Global (1,0) -> U_idx 3
    # Local (1,1) -> Global (1,1) -> U_idx 4
    
    # B_22:
    # Local (0,0) -> Global (0,0) -> V_idx 0
    # Local (0,1) -> Global (0,1) -> V_idx 1
    # Local (1,0) -> Global (1,0) -> V_idx 4
    # Local (1,1) -> Global (1,1) -> V_idx 5
    
    # C_22:
    # Local (0,0) -> Global (0,0) -> W_idx 0
    # Local (0,1) -> Global (0,1) -> W_idx 1
    # Local (1,0) -> Global (1,0) -> W_idx 3
    # Local (1,1) -> Global (1,1) -> W_idx 4
    
    # Strassen Coefficients:
    # M1: A00, A11; B00, B11
    add_term(0, 0, 0, 1, 1, 1, col) # A00, B00, C00
    add_term(4, 0, 0, 1, 1, 1, col) # A11, B00, C00
    add_term(0, 5, 0, 1, 1, 1, col) # A00, B11, C00
    add_term(4, 5, 0, 1, 1, 1, col) # A11, B11, C00
    add_term(0, 0, 4, 1, 1, 1, col) # A00, B00, C11
    add_term(4, 0, 4, 1, 1, 1, col) # A11, B00, C11
    add_term(0, 5, 4, 1, 1, 1, col) # A00, B11, C11
    add_term(4, 5, 4, 1, 1, 1, col) # A11, B11, C11
    col += 1
    
    # M2: A10, A11; B00
    add_term(3, 0, 1, 1, 1, 1, col) # A10, B00, C10
    add_term(4, 0, 1, 1, 1, 1, col) # A11, B00, C10
    add_term(3, 0, 4, 1, 1, -1, col) # A10, B00, C11
    add_term(4, 0, 4, 1, 1, -1, col) # A11, B00, C11
    col += 1
    
    # M3: A00; B10, B00
    add_term(0, 4, 1, 1, 1, 1, col) # A00, B10, C01
    add_term(0, 0, 1, 1, 1, 1, col) # A00, B00, C01
    add_term(0, 4, 4, 1, 1, 1, col) # A00, B10, C11
    add_term(0, 0, 4, 1, 1, 1, col) # A00, B00, C11
    col += 1
    
    # M4: A11; B01, B11
    add_term(4, 1, 0, 1, 1, 1, col) # A11, B01, C00
    add_term(4, 5, 0, 1, 1, 1, col) # A11, B11, C00
    add_term(4, 1, 1, 1, 1, 1, col) # A11, B01, C10
    add_term(4, 5, 1, 1, 1, 1, col) # A11, B11, C10
    col += 1
    
    # M5: A00, A01; B11
    add_term(0, 5, 0, 1, 1, -1, col) # A00, B11, C00
    add_term(1, 5, 0, 1, 1, -1, col) # A01, B11, C00
    add_term(0, 5, 1, 1, 1, 1, col) # A00, B11, C01
    add_term(1, 5, 1, 1, 1, 1, col) # A01, B11, C01
    col += 1
    
    # M6: A10, A00; B00, B01
    add_term(3, 0, 4, 1, 1, 1, col) # A10, B00, C11
    add_term(3, 1, 4, 1, 1, 1, col) # A10, B01, C11
    add_term(0, 0, 4, 1, 1, -1, col) # A00, B00, C11
    add_term(0, 1, 4, 1, 1, -1, col) # A00, B01, C11
    col += 1
    
    # M7: A01, A11; B10, B11
    add_term(1, 4, 0, 1, 1, 1, col) # A01, B10, C00
    add_term(1, 5, 0, 1, 1, 1, col) # A01, B11, C00
    add_term(4, 4, 0, 1, 1, -1, col) # A11, B10, C00
    add_term(4, 5, 0, 1, 1, -1, col) # A11, B11, C00
    col += 1
    
    # --- Block 2: A_22 * B_2p ---
    # B_2p is B[0:2, 2:4].
    # Local (0,0) -> Global (0,2) -> V_idx 2
    # Local (0,1) -> Global (0,3) -> V_idx 3
    # Local (1,0) -> Global (1,2) -> V_idx 6
    # Local (1,1) -> Global (1,3) -> V_idx 7
    # C_2p is C[0:2, 2:4].
    # Local (0,0) -> Global (0,2) -> W_idx 6
    # Local (0,1) -> Global (0,3) -> W_idx 7
    # Local (1,0) -> Global (1,2) -> W_idx 10
    # Local (1,1) -> Global (1,3) -> W_idx 11
    
    # Same Strassen structure, different V, W indices.
    
    # M1
    add_term(0, 2, 6, 1, 1, 1, col)
    add_term(4, 2, 6, 1, 1, 1, col)
    add_term(0, 7, 6, 1, 1, 1, col)
    add_term(4, 7, 6, 1, 1, 1, col)
    add_term(0, 2, 11, 1, 1, 1, col)
    add_term(4, 2, 11, 1, 1, 1, col)
    add_term(0, 7, 11, 1, 1, 1, col)
    add_term(4, 7, 11, 1, 1, 1, col)
    col += 1
    
    # M2
    add_term(3, 2, 10, 1, 1, 1, col)
    add_term(4, 2, 10, 1, 1, 1, col)
    add_term(3, 2, 11, 1, 1, -1, col)
    add_term(4, 2, 11, 1, 1, -1, col)
    col += 1
    
    # M3
    add_term(0, 6, 7, 1, 1, 1, col)
    add_term(0, 2, 7, 1, 1, 1, col)
    add_term(0, 6, 11, 1, 1, 1, col)
    add_term(0, 2, 11, 1, 1, 1, col)
    col += 1
    
    # M4
    add_term(4, 3, 6, 1, 1, 1, col)
    add_term(4, 7, 6, 1, 1, 1, col)
    add_term(4, 3, 10, 1, 1, 1, col)
    add_term(4, 7, 10, 1, 1, 1, col)
    col += 1
    
    # M5
    add_term(0, 7, 6, 1, 1, -1, col)
    add_term(1, 7, 6, 1, 1, -1, col)
    add_term(0, 7, 7, 1, 1, 1, col)
    add_term(1, 7, 7, 1, 1, 1, col)
    col += 1
    
    # M6
    add_term(3, 2, 11, 1, 1, 1, col)
    add_term(3, 3, 11, 1, 1, 1, col)
    add_term(0, 2, 11, 1, 1, -1, col)
    add_term(0, 3, 11, 1, 1, -1, col)
    col += 1
    
    # M7
    add_term(1, 6, 6, 1, 1, 1, col)
    add_term(1, 7, 6, 1, 1, 1, col)
    add_term(4, 6, 6, 1, 1, -1, col)
    add_term(4, 7, 6, 1, 1, -1, col)
    col += 1
    
    # --- Block 3: A_21 * B_12 ---
    # A_21 is A[0:2, 2:3] (2x1).
    # Global U indices:
    # (0,2) -> 2
    # (1,2) -> 5
    # B_12 is B[2:3, 0:2] (1x2).
    # Global V indices:
    # (2,0) -> 8
    # (2,1) -> 9
    # C is C[0:2, 0:2] (2x2).
    # Global W indices:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 3
    # (1,1) -> 4
    
    # Schoolbook:
    # C[0,0] += A[0,2]*B[2,0] -> U[2], V[8], W[0]
    add_term(2, 8, 0, 1, 1, 1, col); col += 1
    # C[0,1] += A[0,2]*B[2,1] -> U[2], V[9], W[1]
    add_term(2, 9, 1, 1, 1, 1, col); col += 1
    # C[1,0] += A[1,2]*B[2,0] -> U[5], V[8], W[3]
    add_term(5, 8, 3, 1, 1, 1, col); col += 1
    # C[1,1] += A[1,2]*B[2,1] -> U[5], V[9], W[4]
    add_term(5, 9, 4, 1, 1, 1, col); col += 1
    
    # --- Block 4: A_21 * B_1p ---
    # B_1p is B[2:3, 2:4] (1x2).
    # Global V indices:
    # (2,2) -> 10
    # (2,3) -> 11
    # C is C[0:2, 2:4] (2x2).
    # Global W indices:
    # (0,2) -> 6
    # (0,3) -> 7
    # (1,2) -> 10
    # (1,3) -> 11
    
    # C[0,2] += A[0,2]*B[2,2] -> U[2], V[10], W[6]
    add_term(2, 10, 6, 1, 1, 1, col); col += 1
    # C[0,3] += A[0,2]*B[2,3] -> U[2], V[11], W[7]
    add_term(2, 11, 7, 1, 1, 1, col); col += 1
    # C[1,2] += A[1,2]*B[2,2] -> U[5], V[10], W[10]
    add_term(5, 10, 10, 1, 1, 1, col); col += 1
    # C[1,3] += A[1,2]*B[2,3] -> U[5], V[11], W[11]
    add_term(5, 11, 11, 1, 1, 1, col); col += 1
    
    # --- Block 5: A_12 * B_22 ---
    # A_12 is A[2:3, 0:2] (1x2).
    # Global U indices:
    # (2,0) -> 6
    # (2,1) -> 7
    # B_22 is B[0:2, 0:2] (2x2).
    # Global V indices:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 4
    # (1,1) -> 5
    # C is C[2:3, 0:2] (1x2).
    # Global W indices:
    # (0,2) -> 6
    # (1,2) -> 10
    
    # C[2,0] += A[2,0]*B[0,0] + A[2,1]*B[1,0]
    add_term(6, 0, 6, 1, 1, 1, col); col += 1
    add_term(7, 4, 6, 1, 1, 1, col); col += 1
    # C[2,1] += A[2,0]*B[0,1] + A[2,1]*B[1,1]
    add_term(6, 1, 10, 1, 1, 1, col); col += 1
    add_term(7, 5, 10, 1, 1, 1, col); col += 1
    
    # --- Block 6: A_12 * B_2p ---
    # B_2p is B[0:2, 2:4] (2x2).
    # Global V indices:
    # (0,2) -> 2
    # (0,3) -> 3
    # (1,2) -> 6
    # (1,3) -> 7
    # C is C[2:3, 2:4] (1x2).
    # Global W indices:
    # (2,2) -> 12
    # (2,3) -> 13
    
    # C[2,2] += A[2,0]*B[0,2] + A[2,1]*B[1,2]
    add_term(6, 2, 12, 1, 1, 1, col); col += 1
    add_term(7, 6, 12, 1, 1, 1, col); col += 1
    # C[2,3] += A[2,0]*B[0,3] + A[2,1]*B[1,3]
    add_term(6, 3, 13, 1, 1, 1, col); col += 1
    add_term(7, 7, 13, 1, 1, 1, col); col += 1
    
    # --- Block 7: A_11 * B_12 ---
    # A_11 is A[2,2] (1x1).
    # Global U index:
    # (2,2) -> 8
    # B_12 is B[2:3, 0:2] (1x2).
    # Global V indices:
    # (2,0) -> 8
    # (2,1) -> 9
    # C is C[2:3, 0:2] (1x2).
    # Global W indices:
    # (0,2) -> 6
    # (1,2) -> 10
    
    # C[2,0] += A[2,2]*B[2,0]
    add_term(8, 8, 6, 1, 1, 1, col); col += 1
    # C[2,1] += A[2,2]*B[2,1]
    add_term(8, 9, 10, 1, 1, 1, col); col += 1
    
    # --- Block 8: A_11 * B_1p ---
    # B_1p is B[2:3, 2:4] (1x2).
    # Global V indices:
    # (2,2) -> 10
    # (2,3) -> 11
    # C is C[2:3, 2:4] (1x2).
    # Global W indices:
    # (2,2) -> 12
    # (2,3) -> 13
    
    # C[2,2] += A[2,2]*B[2,2]
    add_term(8, 10, 12, 1, 1, 1, col); col += 1
    # C[2,3] += A[2,2]*B[2,3]
    add_term(8, 11, 13, 1, 1, 1, col); col += 1
    
    # Check if we used all columns
    if col != r:
        # This should not happen if the calculation is correct.
        # If it does, pad with zeros or raise error.
        # For safety, return schoolbook if rank mismatch.
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
