import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 4, 7
    
    # Helper to create a zero matrix
    def zeros(rows, cols):
        return np.zeros((rows, cols), dtype=float)
    
    # --- Step 1: Construct [2, 2, 2] Strassen Decomposition (Rank 7) ---
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U_s = zeros(4, 7)
    V_s = zeros(4, 7)
    W_s = zeros(4, 7)
    
    # M1: (A11+A22)(B11+B22) -> C11(+1), C22(+1)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21(+1), C22(-1)
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1
    
    # M3: A11(B12-B22) -> C12(+1), C22(+1)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = 1
    
    # M4: A22(B21-B11) -> C11(+1), C21(+1)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 1
    
    # M5: (A11+A12)B22 -> C11(-1), C12(+1)
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C22(+1)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C11(+1)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # --- Step 2: Construct [2, 2, 3] Schoolbook Decomposition (Rank 12) ---
    # U shape (4, 12), V shape (6, 12), W shape (6, 12)
    U_t = zeros(4, 12)
    V_t = zeros(6, 12)
    W_t = zeros(6, 12)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_t[i * 2 + j, col] = 1
                V_t[j * 3 + k, col] = 1
                W_t[k * 2 + i, col] = 1
                col += 1
                
    # --- Step 3: Construct [2, 2, 4] Decomposition (Rank 14) ---
    # Combines two [2, 2, 2] blocks.
    # Block 1: k=0,1. Block 2: k=2,3.
    U_q = zeros(4, 14)
    V_q = zeros(8, 14) # 2 rows * 4 cols
    W_q = zeros(8, 14) # 4 cols * 2 rows
    
    # Block 1 (k=0,1)
    U_q[:, 0:7] = U_s
    V_q[0:4, 0:7] = V_s
    W_q[0:4, 0:7] = W_s
    
    # Block 2 (k=2,3)
    # B indices shift by 2 (since p=4, j*p+k. j=0 -> 2,3. j=1 -> 6,7).
    # Wait, V index is j*p + k. For [2,2,2], p=2.
    # For [2,2,4], p=4.
    # Block 2 uses k=2,3.
    # B index for j=0, k=2 is 0*4+2=2. j=1, k=2 is 1*4+2=6.
    # In V_s, indices are 0..3.
    # We need to map V_s indices to V_q indices.
    # V_s row r corresponds to B index r in [2,2,2].
    # In [2,2,4], B index = j*4 + k.
    # For Block 2, k is 2 or 3.
    # If V_s has 1 at index 0 (j=0, k=0), in Block 2 it should be at j=0, k=2 -> index 2.
    # If V_s has 1 at index 1 (j=0, k=1), in Block 2 it should be at j=0, k=3 -> index 3.
    # If V_s has 1 at index 2 (j=1, k=0), in Block 2 it should be at j=1, k=2 -> index 6.
    # If V_s has 1 at index 3 (j=1, k=1), in Block 2 it should be at j=1, k=3 -> index 7.
    
    # So we copy V_s to V_q rows 2,3,6,7.
    V_q[2, 7:14] = V_s[0, 0:7]
    V_q[3, 7:14] = V_s[1, 0:7]
    V_q[6, 7:14] = V_s[2, 0:7]
    V_q[7, 7:14] = V_s[3, 0:7]
    
    # W indices for Block 2:
    # W index = k*n + i. n=2.
    # Block 1: k=0,1. W indices 0..3.
    # Block 2: k=2,3.
    # If k=2, i=0 -> 4. i=1 -> 5.
    # If k=3, i=0 -> 6. i=1 -> 7.
    # W_s indices 0..3 correspond to k=0,1.
    # We need to map W_s indices to W_q indices 4..7.
    # W_s[0] (k=0, i=0) -> W_q[4]
    # W_s[1] (k=0, i=1) -> W_q[5]
    # W_s[2] (k=1, i=0) -> W_q[6]
    # W_s[3] (k=1, i=1) -> W_q[7]
    
    W_q[4, 7:14] = W_s[0, 0:7]
    W_q[5, 7:14] = W_s[1, 0:7]
    W_q[6, 7:14] = W_s[2, 0:7]
    W_q[7, 7:14] = W_s[3, 0:7]
    
    U_q[:, 7:14] = U_s
    
    # --- Step 4: Construct [2, 2, 7] Decomposition (Rank 26) ---
    # Combines [2, 2, 4] (Rank 14) and [2, 2, 3] (Rank 12).
    U_sev = zeros(4, 26)
    V_sev = zeros(14, 26) # 2 rows * 7 cols
    W_sev = zeros(14, 26) # 7 cols * 2 rows
    
    # Part 1: [2, 2, 4]
    U_sev[:, 0:14] = U_q
    # V_q indices (0..7) map to V_sev indices (0..7) because k=0..3.
    V_sev[0:8, 0:14] = V_q
    # W_q indices (0..7) map to W_sev indices (0..7) because k=0..3.
    W_sev[0:8, 0:14] = W_q
    
    # Part 2: [2, 2, 3]
    U_sev[:, 14:26] = U_t
    # V_t indices (0..5) correspond to j=0..1, k=0..2 in [2,2,3].
    # In [2,2,7], these correspond to k=4..6.
    # B index = j*7 + k.
    # j=0, k=4 -> 4. j=0, k=5 -> 5. j=0, k=6 -> 6.
    # j=1, k=4 -> 11. j=1, k=5 -> 12. j=1, k=6 -> 13.
    # V_t row 0 (j=0, k=0) -> V_sev row 4
    # V_t row 1 (j=0, k=1) -> V_sev row 5
    # V_t row 2 (j=0, k=2) -> V_sev row 6
    # V_t row 3 (j=1, k=0) -> V_sev row 11
    # V_t row 4 (j=1, k=1) -> V_sev row 12
    # V_t row 5 (j=1, k=2) -> V_sev row 13
    
    V_sev[4, 14:26] = V_t[0, 0:12]
    V_sev[5, 14:26] = V_t[1, 0:12]
    V_sev[6, 14:26] = V_t[2, 0:12]
    V_sev[11, 14:26] = V_t[3, 0:12]
    V_sev[12, 14:26] = V_t[4, 0:12]
    V_sev[13, 14:26] = V_t[5, 0:12]
    
    # W_t indices (0..5) correspond to k=0..2, i=0..1 in [2,2,3].
    # In [2,2,7], these correspond to k=4..6.
    # W index = k*2 + i.
    # k=4, i=0 -> 8. k=4, i=1 -> 9.
    # k=5, i=0 -> 10. k=5, i=1 -> 11.
    # k=6, i=0 -> 12. k=6, i=1 -> 13.
    # W_t row 0 (k=0, i=0) -> W_sev row 8
    # W_t row 1 (k=0, i=1) -> W_sev row 9
    # W_t row 2 (k=1, i=0) -> W_sev row 10
    # W_t row 3 (k=1, i=1) -> W_sev row 11
    # W_t row 4 (k=2, i=0) -> W_sev row 12
    # W_t row 5 (k=2, i=1) -> W_sev row 13
    
    W_sev[8, 14:26] = W_t[0, 0:12]
    W_sev[9, 14:26] = W_t[1, 0:12]
    W_sev[10, 14:26] = W_t[2, 0:12]
    W_sev[11, 14:26] = W_t[3, 0:12]
    W_sev[12, 14:26] = W_t[4, 0:12]
    W_sev[13, 14:26] = W_t[5, 0:12]
    
    # --- Step 5: Construct [2, 4, 7] Decomposition (Rank 52) ---
    # Combines two [2, 2, 7] blocks.
    # Block 1: j=0,1. Block 2: j=2,3.
    U_final = zeros(8, 52) # 2 rows * 4 cols
    V_final = zeros(28, 52) # 4 rows * 7 cols
    W_final = zeros(14, 52) # 7 cols * 2 rows
    
    # Block 1 (j=0,1)
    # U_sev indices 0..3 correspond to i=0..1, j=0..1 in [2,2,7].
    # In [2,4,7], A index = i*4 + j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 4. i=1, j=1 -> 5.
    # So U_sev row 0 -> U_final row 0.
    # U_sev row 1 -> U_final row 1.
    # U_sev row 2 -> U_final row 4.
    # U_sev row 3 -> U_final row 5.
    
    U_final[0, 0:26] = U_sev[0, 0:26]
    U_final[1, 0:26] = U_sev[1, 0:26]
    U_final[4, 0:26] = U_sev[2, 0:26]
    U_final[5, 0:26] = U_sev[3, 0:26]
    
    # V_sev indices 0..13 correspond to j=0..1, k=0..6.
    # In [2,4,7], B index = j*7 + k.
    # j=0, k=0..6 -> 0..6.
    # j=1, k=0..6 -> 7..13.
    # So V_sev rows 0..6 -> V_final rows 0..6.
    # V_sev rows 7..13 -> V_final rows 7..13.
    
    V_final[0:14, 0:26] = V_sev
    
    # W_sev indices 0..13 correspond to k=0..6, i=0..1.
    # In [2,4,7], W index = k*2 + i.
    # Same indices.
    
    W_final[0:14, 0:26] = W_sev
    
    # Block 2 (j=2,3)
    # U_sev rows 0..3 correspond to i=0..1, j=0..1 in [2,2,7].
    # In [2,4,7], these correspond to j=2,3.
    # A index = i*4 + j.
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    
    U_final[2, 26:52] = U_sev[0, 0:26]
    U_final[3, 26:52] = U_sev[1, 0:26]
    U_final[6, 26:52] = U_sev[2, 0:26]
    U_final[7, 26:52] = U_sev[3, 0:26]
    
    # V_sev rows 0..13 correspond to j=0..1, k=0..6.
    # In [2,4,7], these correspond to j=2,3.
    # B index = j*7 + k.
    # j=2, k=0..6 -> 14..20.
    # j=3, k=0..6 -> 21..27.
    # V_sev rows 0..6 (j=0) -> V_final rows 14..20.
    # V_sev rows 7..13 (j=1) -> V_final rows 21..27.
    
    V_final[14:21, 26:52] = V_sev[0:7, 0:26]
    V_final[21:28, 26:52] = V_sev[7:14, 0:26]
    
    # W indices are the same.
    W_final[0:14, 26:52] = W_sev
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

# EVOLVE-BLOCK-END
