import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    Strategy: Apply Strassen's algorithm to 2x2 blocks where possible.
    The 3x4x5 multiplication can be decomposed into blocks.
    Since n=3, m=4, p=5, we can partition the inner dimension (m=4) into two 2s.
    The output C is 3x5.
    
    We can view this as multiplying a 3x4 matrix A and 4x5 matrix B.
    Let's split A into A_top (3x2) and A_bot (3x2).
    Let's split B into B_left (2x5) and B_right (2x5).
    
    Actually, Strassen works on 2x2 matrices. 
    We can group the rows of C (3 rows) into a 2x2 block (rows 0,1) and a 1x2 block (row 2)?
    Or group columns?
    
    Let's try to apply Strassen to the 2x2 sub-problem involving A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2].
    And A[0:2, 2:4], B[2:4, 0:2] -> C[0:2, 0:2].
    This is getting complicated because of the rectangular shapes.
    
    Alternative: 
    The tensor rank of M_{3,4,5} is known to be 56? No, let's check known results.
    Schoolbook is 60.
    Strassen reduces 2x2x2 from 8 to 7.
    If we can find four disjoint 2x2x2 sub-tensors, we save 4.
    Can we?
    m=4. We can split m into 0,1 and 2,3.
    n=3. We can split n into 0,1 and 2.
    p=5. We can split p into 0,1 and 2,3,4.
    
    Consider the sub-tensor for i in {0,1}, j in {0,1}, k in {0,1}. This is a 2x2x2 block. Rank 7 instead of 8.
    Consider the sub-tensor for i in {0,1}, j in {2,3}, k in {0,1}. This is a 2x2x2 block. Rank 7 instead of 8.
    Consider the sub-tensor for i in {0,1}, j in {0,1}, k in {2,3}. This is a 2x2x2 block. Rank 7 instead of 8.
    Consider the sub-tensor for i in {0,1}, j in {2,3}, k in {2,3}. This is a 2x2x2 block. Rank 7 instead of 8.
    
    That accounts for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    The remaining parts are:
    1. i=2, j in {0,1,2,3}, k in {0,1,2,3,4}. (4*5 = 20 terms)
    2. i in {0,1}, j in {0,1,2,3}, k=4. (2*4 = 8 terms)
    
    Total terms = 4 * 7 (Strassen) + 20 + 8 = 28 + 28 = 56.
    
    Let's verify the Strassen application.
    For a 2x2 block A (indices i,j), B (indices j,k), C (indices i,k).
    The 7 products are:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B10 - B11)
    M4 = A11 * (B01 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need to map these to the global indices.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}.
    Block 2: i in {0,1}, j in {2,3}, k in {0,1}.
    Block 3: i in {0,1}, j in {0,1}, k in {2,3}.
    Block 4: i in {0,1}, j in {2,3}, k in {2,3}.
    
    The remaining terms are just schoolbook.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS
    rank = 56
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_base, j_base, k_base, rank_offset):
        """
        Adds Strassen terms for a 2x2 block starting at (i_base, j_base, k_base).
        i_base, j_base, k_base are 0-indexed bases.
        The block covers i in {i_base, i_base+1}, j in {j_base, j_base+1}, k in {k_base, k_base+1}.
        """
        # Indices for A (U)
        u00 = i_base * m + j_base
        u01 = i_base * m + (j_base + 1)
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + (j_base + 1)
        
        # Indices for B (V)
        v00 = j_base * p + k_base
        v01 = j_base * p + (k_base + 1)
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + (k_base + 1)
        
        # Indices for C (W)
        # W index is k * n + i
        w00 = k_base * n + i_base
        w01 = k_base * n + (i_base + 1) # Wait, C01 is row 0, col 1. So i=0, k=1.
        # C is n x p. C[i, k].
        # W[k * n + i].
        # C00: i=0, k=0 -> w00 = 0*n + 0
        # C01: i=0, k=1 -> w01 = 1*n + 0
        # C10: i=1, k=0 -> w10 = 0*n + 1
        # C11: i=1, k=1 -> w11 = 1*n + 1
        
        w00_idx = k_base * n + i_base
        w01_idx = (k_base + 1) * n + i_base
        w10_idx = k_base * n + (i_base + 1)
        w11_idx = (k_base + 1) * n + (i_base + 1)
        
        # M1: (A00 + A11) * (B00 + B11)
        # U: u00 += 1, u11 += 1
        # V: v00 += 1, v11 += 1
        # W: w00 += 1, w11 += 1
        r = rank_offset
        U[u00][r] += 1
        U[u11][r] += 1
        V[v00][r] += 1
        V[v11][r] += 1
        W[w00_idx][r] += 1
        W[w11_idx][r] += 1
        
        # M2: (A10 + A11) * B00
        # U: u10 += 1, u11 += 1
        # V: v00 += 1
        # W: w10 += 1
        r += 1
        U[u10][r] += 1
        U[u11][r] += 1
        V[v00][r] += 1
        W[w10_idx][r] += 1
        
        # M3: A00 * (B10 - B11)
        # U: u00 += 1
        # V: v10 += 1, v11 -= 1
        # W: w01 += 1, w11 += 1
        r += 1
        U[u00][r] += 1
        V[v10][r] += 1
        V[v11][r] -= 1
        W[w01_idx][r] += 1
        W[w11_idx][r] += 1
        
        # M4: A11 * (B01 - B00)
        # U: u11 += 1
        # V: v01 += 1, v00 -= 1
        # W: w00 += 1, w10 += 1
        r += 1
        U[u11][r] += 1
        V[v01][r] += 1
        V[v00][r] -= 1
        W[w00_idx][r] += 1
        W[w10_idx][r] += 1
        
        # M5: (A00 + A01) * B11
        # U: u00 += 1, u01 += 1
        # V: v11 += 1
        # W: w00 -= 1, w01 += 1
        r += 1
        U[u00][r] += 1
        U[u01][r] += 1
        V[v11][r] += 1
        W[w00_idx][r] -= 1
        W[w01_idx][r] += 1
        
        # M6: (A10 - A00) * (B00 + B01)
        # U: u10 += 1, u00 -= 1
        # V: v00 += 1, v01 += 1
        # W: w11 += 1
        r += 1
        U[u10][r] += 1
        U[u00][r] -= 1
        V[v00][r] += 1
        V[v01][r] += 1
        W[w11_idx][r] += 1
        
        # M7: (A01 - A11) * (B10 + B11)
        # U: u01 += 1, u11 -= 1
        # V: v10 += 1, v11 += 1
        # W: w00 += 1
        r += 1
        U[u01][r] += 1
        U[u11][r] -= 1
        V[v10][r] += 1
        V[v11][r] += 1
        W[w00_idx][r] += 1
        
        return r + 1
    
    col = 0
    
    # Apply Strassen to the four 2x2x2 blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    col = add_strassen_block(0, 2, 0, col)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    col = add_strassen_block(0, 2, 2, col)
    
    # Now handle the remaining terms with schoolbook
    # Remaining i: {2}
    # Remaining j: {0, 1, 2, 3}
    # Remaining k: {0, 1, 2, 3, 4}
    
    # Also, for i in {0, 1}, k = 4 is not covered by the blocks above (which covered k in {0,1,2,3}).
    # So we need:
    # 1. i=2, all j, all k
    # 2. i in {0,1}, all j, k=4
    
    # Part 1: i=2
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # Part 2: i in {0,1}, k=4
    for i in range(2):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    # Check if col == rank
    # Rank should be 56.
    # 4 blocks * 7 = 28
    # i=2: 4 * 5 = 20
    # i in {0,1}, k=4: 2 * 4 = 8
    # Total = 28 + 20 + 8 = 56.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
