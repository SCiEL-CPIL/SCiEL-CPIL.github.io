"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen decomposition for M2 (rank 7)
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A indices: 0,1 (row 0), 2,3 (row 1)
    U2[0,0] = 1; U2[1,0] = 0; U2[2,0] = 1; U2[3,0] = 0 # Wait, A11 is (0,0),(0,1). A22 is (1,0),(1,1).
    # Let's use explicit indices
    # A = [[A00, A01], [A10, A11]]
    # Indices: 0:00, 1:01, 2:10, 3:11
    U2[0,0] = 1; U2[1,0] = 0; U2[2,0] = 1; U2[3,0] = 0 # A00+A11? No.
    # A11 in Strassen notation is top-left.
    # My indexing: 0=00, 1=01, 2=10, 3=11.
    # Strassen A11 -> index 0. A12 -> index 1. A21 -> index 2. A22 -> index 3.
    
    # M1: (A11+A22) -> indices 0, 3
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    # C11 += M1, C22 += M1
    W2[0,0] = 1; W2[3,0] = 1
    
    # M2: (A21+A22)B11 -> indices 2,3 for A; 0 for B
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    # C21 += M2, C22 -= M2
    W2[2,1] = 1; W2[3,1] = -1
    
    # M3: A11(B12-B22) -> index 0 for A; 1,3 for B
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    # C12 += M3, C22 += M3
    W2[1,2] = 1; W2[3,2] = 1
    
    # M4: A22(B21-B11) -> index 3 for A; 2,0 for B
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    # C11 += M4, C21 += M4
    W2[0,3] = 1; W2[2,3] = 1
    
    # M5: (A11+A12)B22 -> indices 0,1 for A; 3 for B
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    # C11 -= M5, C12 += M5
    W2[0,4] = -1; W2[1,4] = 1
    
    # M6: (A21-A11)(B11+B12) -> indices 2,0 for A; 0,1 for B
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    # C22 += M6
    W2[3,5] = 1
    
    # M7: (A12-A22)(B21+B22) -> indices 1,3 for A; 2,3 for B
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    # C11 += M7
    W2[0,6] = 1

    # 2. Construct M4 decomposition (rank 49) using Kronecker product
    # U4 = U2 kron U2, etc.
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # 3. Construct M5 decomposition (rank 95)
    # Dimensions: U (25x95), V (25x95), W (25x95)
    # Note: V corresponds to B (5x5), W to C (5x5).
    # Indices 0-15 are top-left 4x4 block.
    # Indices 16-19 are top-right 4x1 block (rows 0-3, col 4).
    # Indices 20-23 are bottom-left 1x4 block (row 4, cols 0-3).
    # Index 24 is bottom-right 1x1 block (row 4, col 4).
    
    U5 = np.zeros((25, 95))
    V5 = np.zeros((25, 95))
    W5 = np.zeros((25, 95))
    
    col = 0
    
    # Block 1: A11 B11 (M4, rank 49)
    # A indices 0-15, B indices 0-15, C indices 0-15
    U5[0:16, col:col+49] = U4
    V5[0:16, col:col+49] = V4
    W5[0:16, col:col+49] = W4
    col += 49
    
    # Block 2: A12 B21 (Outer product, rank 1)
    # A indices 16-19 (col 4, rows 0-3)
    # B indices 20-23 (row 4, cols 0-3)
    # C indices 0-15 (rows 0-3, cols 0-3)
    # C_ij = A_i4 B_4j
    # U: 1 at A indices
    # V: 1 at B indices
    # W: 1 at C indices
    # This is rank 1?
    # Sum over k? No, k is fixed?
    # A12 is 4x1. B21 is 1x4.
    # Product is 4x4.
    # C_ij = A_i,4 * B_4,j.
    # This is a single product of linear forms?
    # A_part = vector of A_i,4.
    # B_part = vector of B_4,j.
    # C_part = outer product.
    # Rank 1.
    U5[16:20, col] = 1
    V5[20:24, col] = 1
    W5[0:16, col] = 1 # Wait, W is 25x1. C indices 0-15 are 16 elements.
    # Outer product of u (4x1) and v (1x4) is 4x4.
    # Flattened, it's 16 elements.
    # So W5[0:16, col] = 1 is correct if the structure matches.
    # Yes, C_ij = u_i v_j.
    col += 1
    
    # Block 3: A11 B12 (M4,4,1, rank 16)
    # A indices 0-15
    # B indices 16-19 (col 4, rows 0-3)
    # C indices 16-19 (col 4, rows 0-3)
    # C_i,4 = sum_k A_i,k B_k,4
    # Schoolbook rank 16.
    for i in range(4):
        for k in range(4):
            U5[i*4+k, col] = 1
            V5[k*4+4, col] = 1 # B index: row k, col 4 -> k*5+4?
            # Wait, B is 5x5.
            # B12 is top-right 4x1.
            # B indices for this block:
            # Row 0, Col 4 -> 4
            # Row 1, Col 4 -> 9
            # Row 2, Col 4 -> 14
            # Row 3, Col 4 -> 19
            # So V5 indices are 4, 9, 14, 19.
            V5[k*5+4, col] = 1
            W5[i*5+4, col] = 1 # C index: row i, col 4 -> i*5+4
            col += 1
    col -= 12 # We added 16, but loop ran 4x4=16 times. Wait.
    # The loop adds 16 columns.
    # So col is now col_start + 16.
    
    # Block 4: A12 B22 (M1,4,1, rank 4)
    # A indices 16-19
    # B index 24
    # C indices 16-19
    # C_i,4 = A_i,4 B_4,4
    for i in range(4):
        U5[i*5+4, col] = 1
        V5[24, col] = 1
        W5[i*5+4, col] = 1
        col += 1
        
    # Block 5: A21 B11 (M1,4,4, rank 16)
    # A indices 20-23
    # B indices 0-15
    # C indices 20-23
    # C_4,j = sum_k A_4,k B_k,j
    for j in range(4):
        for k in range(4):
            U5[4*5+k, col] = 1 # A index: row 4, col k -> 20+k
            V5[k*5+j, col] = 1
            W5[4*5+j, col] = 1
            col += 1
            
    # Block 6: A22 B21 (M1,1,4, rank 4)
    # A index 24
    # B indices 20-23
    # C indices 20-23
    # C_4,j = A_4,4 B_4,j
    for j in range(4):
        U5[24, col] = 1
        V5[4*5+j, col] = 1
        W5[4*5+j, col] = 1
        col += 1
        
    # Block 7: A21 B12 (M1,4,1, rank 4)
    # A indices 20-23
    # B indices 16-19
    # C index 24
    # C_4,4 = sum_k A_4,k B_k,4
    for k in range(4):
        U5[4*5+k, col] = 1
        V5[k*5+4, col] = 1
        W5[24, col] = 1
        col += 1
        
    # Block 8: A22 B22 (M1,1, rank 1)
    # A index 24
    # B index 24
    # C index 24
    U5[24, col] = 1
    V5[24, col] = 1
    W5[24, col] = 1
    col += 1
    
    # Convert to lists
    U_list = U5.tolist()
    V_list = V5.tolist()
    W_list = W5.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

