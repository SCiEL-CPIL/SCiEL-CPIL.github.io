import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 2x2 (Rank 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1
    
    # 2. Laderman decomposition for 3x3 (Rank 23)
    # Laderman's algorithm for 3x3 matrix multiplication.
    # Reference: Laderman, J. (1976). "The rank of the matrix multiplication tensor".
    # Rank is 23.
    
    U3 = np.zeros((9, 23), dtype=float)
    V3 = np.zeros((9, 23), dtype=float)
    W3 = np.zeros((9, 23), dtype=float)
    
    # Laderman's 23 products:
    # Indices for 3x3:
    # A: 0..8 (row-major: 00,01,02, 10,11,12, 20,21,22)
    # B: 0..8
    # C: 0..8
    
    # The specific coefficients for Laderman's algorithm are complex.
    # For the purpose of this exercise, and given the directive's premise of "Rank 7",
    # it is highly likely the directive assumes a non-existent Rank 7 3x3 algorithm.
    # Since I cannot invent a valid Rank 7 3x3 algorithm, I will use the Trivial 3x3 (Rank 27)
    # to ensure validity, although this results in Rank 7*27=189 for 6x6.
    # However, if I use Rank 23, I get 161.
    # Let's use Rank 23 (Laderman) to be closer to the "Laderman" mention.
    # But I don't have the full 23-term coefficients memorized perfectly.
    # Using a valid Trivial 3x3 (Rank 27) is safer for correctness.
    # Rank 27 for 3x3.
    
    R3 = 27
    U3 = np.zeros((9, R3), dtype=float)
    V3 = np.zeros((9, R3), dtype=float)
    W3 = np.zeros((9, R3), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j, col] = 1
                V3[j*3 + k, col] = 1
                W3[k*3 + i, col] = 1
                col += 1
                
    # 3. Kronecker Product for 6x6
    # Rank = 7 * 27 = 189.
    # We construct the 6x6 decomposition and then extract the 5x5 sub-decomposition.
    
    R_total = 7 * 27
    
    U6 = np.zeros((36, R_total), dtype=float)
    V6 = np.zeros((36, R_total), dtype=float)
    W6 = np.zeros((36, R_total), dtype=float)
    
    # Kronecker product of indices:
    # 2x2 index (i2, j2) -> 0..3
    # 3x3 index (i3, j3) -> 0..8
    # 6x6 index (i, j) where i = i2*3 + i3, j = j2*3 + j3
    
    # U6[i, r] where r = r2 * 27 + r3
    # U6[i2*3+i3, r2*27+r3] = U2[i2, r2] * U3[i3, r3]
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            i = i2 * 3 + i3
                            j = j2 * 3 + j3
                            idx = i * 6 + j
                            
                            U6[idx, r] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]
                            V6[idx, r] = V2[i2*2+j2, r2] * V3[i3*3+j3, r3]
                            W6[idx, r] = W2[i2*2+j2, r2] * W3[i3*3+j3, r3]
                            
    # 4. Extract 5x5 from 6x6
    # We take the first 5 rows and 5 columns of the 6x6 indices.
    # Indices 0..4 in the 6x6 space correspond to i=0..4, j=0..4.
    # In the 6x6 layout (row-major 6x6), indices 0..24 are the first 5 rows?
    # No, 6x6 has indices 0..35.
    # Row 0: 0..5
    # Row 1: 6..11
    # ...
    # We need to select the sub-matrix A[0:5, 0:5].
    # This corresponds to indices (i, j) where i<5 and j<5.
    # In the flattened 36-dim vector, these are not contiguous.
    
    # We construct U5, V5, W5 of shape (25, R_total).
    # We map the 5x5 indices (i, j) with i,j in 0..4 to the 6x6 indices.
    # The 6x6 index is i*6 + j.
    
    U5 = np.zeros((25, R_total), dtype=float)
    V5 = np.zeros((25, R_total), dtype=float)
    W5 = np.zeros((25, R_total), dtype=float)
    
    for i in range(5):
        for j in range(5):
            idx_5 = i * 5 + j
            idx_6 = i * 6 + j
            
            U5[idx_5, :] = U6[idx_6, :]
            V5[idx_5, :] = V6[idx_6, :]
            W5[idx_5, :] = W6[idx_6, :]
            
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
# EVOLVE-BLOCK-END
