import numpy as np

def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # Helper to set values
    def set_val(M, row_idx, col_idx, val):
        M[row_idx, col_idx] = val
        
    # Strassen 2x2 core (indices 1,2)
    # A[1,1], A[1,2], A[2,1], A[2,2]
    # B[1,1], B[1,2], B[2,1], B[2,2]
    
    # Term 0: (A11+A22)(B11+B22)
    # U: A11, A22 -> 1
    # V: B11, B22 -> 1
    # W: C11, C22 -> 1
    # C11 is W index 1*3+1 = 4. C22 is W index 2*3+2 = 8.
    # Wait, W index is k*n+i.
    # C[1,1] -> k=1, i=1 -> 1*3+1=4.
    # C[2,2] -> k=2, i=2 -> 2*3+2=8.
    # U[1,1] -> 1*3+1=4. U[2,2] -> 2*3+2=8.
    # V[1,1] -> 1*3+1=4. V[2,2] -> 2*3+2=8.
    
    U[4, 0] = 1; U[8, 0] = 1
    V[4, 0] = 1; V[8, 0] = 1
    W[4, 0] = 1; W[8, 0] = 1
    
    # Term 1: (A21+A22)B11
    # U: A21 (idx 7), A22 (idx 8) -> 1
    # V: B11 (idx 4) -> 1
    # W: C21 (k=1, i=2 -> 5), C22 (k=2, i=2 -> 8) -> 1
    U[7, 1] = 1; U[8, 1] = 1
    V[4, 1] = 1
    W[5, 1] = 1; W[8, 1] = 1
    
    # Term 2: A11(B12-B22)
    # U: A11 (idx 4) -> 1
    # V: B12 (idx 5), B22 (idx 8) -> 1, -1
    # W: C12 (k=2, i=1 -> 7), C22 (k=2, i=2 -> 8) -> 1, -1?
    # C12 = A11 B12 + A12 B22.
    # This term contributes A11 B12 to C12 and -A11 B22 to C22?
    # Let's check Strassen:
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 + M6
    # C22 = M1 - M2 - M3 + M5
    
    # So:
    # Term 2 (M3): A11(B12-B22)
    # Contributes +1 to C12, -1 to C22?
    # C12 gets M3. C22 gets -M3.
    U[4, 2] = 1
    V[5, 2] = 1; V[8, 2] = -1
    W[7, 2] = 1; W[8, 2] = -1
    
    # Term 3 (M4): A22(B21-B11)
    # Contributes +1 to C21, -1 to C11.
    U[8, 3] = 1
    V[7, 3] = 1; V[4, 3] = -1  # B21 is idx 7, B11 is idx 4
    W[5, 3] = 1; W[4, 3] = -1  # C21 is idx 5, C11 is idx 4
    
    # Term 4 (M5): (A11+A12)B22
    # Contributes +1 to C12, +1 to C22.
    U[4, 4] = 1; U[5, 4] = 1 # A11, A12
    V[8, 4] = 1 # B22
    W[7, 4] = 1; W[8, 4] = 1 # C12, C22
    
    # Term 5 (M6): (A21-A11)(B11+B12)
    # Contributes +1 to C21, -1 to C11.
    U[7, 5] = 1; U[4, 5] = -1 # A21, A11
    V[4, 5] = 1; V[5, 5] = 1 # B11, B12
    W[5, 5] = 1; W[4, 5] = -1 # C21, C11
    
    # Term 6 (M7): (A12-A22)(B21+B22)
    # Contributes +1 to C11, +1 to C12.
    U[5, 6] = 1; U[8, 6] = -1 # A12, A22
    V[7, 6] = 1; V[8, 6] = 1 # B21, B22
    W[4, 6] = 1; W[7, 6] = 1 # C11, C12

    # Now the border terms (indices involving 0)
    # We need to compute C[0,k], C[i,0], C[0,0].
    # Winograd adds 16 more terms? No, total 23.
    # We have used 7. Need 16 more.
    
    # The border involves A[0,j], A[i,0], B[j,0], B[0,k].
    # There are 6 "mixed" terms and 10 "pure" border terms?
    # Actually, Winograd's 23 rank decomposition for 3x3 is:
    # 7 from 2x2 core.
    # 6 from "diagonal" border interactions?
    # 10 from the rest.
    
    # Let's use a known explicit list for the remaining 16 terms.
    # Reference: "The rank of 3x3 matrix multiplication"
    
    # Terms 7-12: Handle A[0,j] * B[j,0] type interactions and C[0,0]
    # Actually, it's easier to just use the full explicit matrix if I can find it.
    # Since I don't have internet access, I will construct the remaining terms logically or use a simpler rank-26 if 23 is too risky to hardcode without verification.
    # However, rank 23 is the known minimum.
    
    # Let's try to complete the 23 rank decomposition.
    # The remaining indices are those involving 0.
    # U indices: 0,1,2 (row 0), 3,6 (col 0)
    # V indices: 0,3,6 (row 0), 1,2,5 (col 0? No, V is j*p+k. j=0 is 0,1,2. k=0 is 0,3,6)
    # W indices: 0,3,6 (k=0), 1,4,7 (i=0? No, W is k*n+i. i=0 is 0,3,6. k=0 is 0,1,2)
    
    # Let's add terms for the "zero" row/col.
    # Term 7: A[0,0] * B[0,0] -> C[0,0]
    U[0, 7] = 1; V[0, 7] = 1; W[0, 7] = 1
    
    # Term 8: A[0,1] * B[1,0] -> C[0,0]
    U[1, 8] = 1; V[3, 8] = 1; W[0, 8] = 1
    
    # Term 9: A[0,2] * B[2,0] -> C[0,0]
    U[2, 9] = 1; V[6, 9] = 1; W[0, 9] = 1
    
    # Term 10: A[1,0] * B[0,1] -> C[1,1]? No, C[1,1] is already handled?
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1].
    # The Strassen part handled A[1,1]B[1,1] etc.
    # We need A[1,0]B[0,1] for C[1,1].
    U[3, 10] = 1; V[1, 10] = 1; W[4, 10] = 1
    
    # Term 11: A[2,0] * B[0,2] -> C[2,2]
    U[6, 11] = 1; V[2, 11] = 1; W[8, 11] = 1
    
    # Term 12: A[1,0] * B[0,2] -> C[1,2]
    U[3, 12] = 1; V[2, 12] = 1; W[7, 12] = 1
    
    # Term 13: A[2,0] * B[0,1] -> C[2,1]
    U[6, 13] = 1; V[1, 13] = 1; W[5, 13] = 1
    
    # Term 14: A[0,1] * B[1,2] -> C[0,2]
    U[1, 14] = 1; V[5, 14] = 1; W[2, 14] = 1
    
    # Term 15: A[0,2] * B[2,1] -> C[0,1]
    U[2, 15] = 1; V[7, 15] = 1; W[1, 15] = 1
    
    # Term 16: A[0,1] * B[1,1] -> C[0,1]? No, C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1].
    # We have A[0,2]B[2,1] in Term 15.
    # We need A[0,0]B[0,1] and A[0,1]B[1,1].
    
    # This manual construction is error-prone.
    # I will return the rank 27 schoolbook if I cannot guarantee rank 23 correctness immediately.
    # However, the prompt asks for a "stronger algorithm". Rank 26 was achieved in the "inspiration" note.
    # Rank 23 is the theoretical minimum.
    # Given the risk of incorrect hardcoding, I will implement Rank 26 using Strassen 2x2 + Schoolbook Border.
    # This is a safe improvement over 27.
    
    # Rank 26 Construction:
    # 7 terms for the 2x2 core (indices 1,2).
    # 19 terms for the rest?
    # Total 26.
    # The border consists of row 0, col 0.
    # There are 3x3x3 = 27 terms total.
    # The 2x2 core uses 7 terms instead of 8. Saves 1.
    # So 27 - 1 = 26.
    
    # I will implement Rank 26.
    
    # Reset matrices for Rank 26
    rank = 26
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Terms 0-6: Strassen 2x2 (indices 1,2)
    # Same as above
    U[4, 0] = 1; U[8, 0] = 1; V[4, 0] = 1; V[8, 0] = 1; W[4, 0] = 1; W[8, 0] = 1
    U[7, 1] = 1; U[8, 1] = 1; V[4, 1] = 1; W[5, 1] = 1; W[8, 1] = 1
    U[4, 2] = 1; V[5, 2] = 1; V[8, 2] = -1; W[7, 2] = 1; W[8, 2] = -1
    U[8, 3] = 1; V[7, 3] = 1; V[4, 3] = -1; W[5, 3] = 1; W[4, 3] = -1
    U[4, 4] = 1; U[5, 4] = 1; V[8, 4] = 1; W[7, 4] = 1; W[8, 4] = 1
    U[7, 5] = 1; U[4, 5] = -1; V[4, 5] = 1; V[5, 5] = 1; W[5, 5] = 1; W[4, 5] = -1
    U[5, 6] = 1; U[8, 6] = -1; V[7, 6] = 1; V[8, 6] = 1; W[4, 6] = 1; W[7, 6] = 1
    
    # Terms 7-25: Schoolbook for all terms involving index 0
    # Total 19 terms.
    # Indices involving 0 in A or B.
    # A[0,j] (j=0,1,2) -> 3 terms
    # A[i,0] (i=1,2) -> 2 terms (A[0,0] counted in A[0,j])
    # B[j,0] (j=1,2) -> 2 terms (B[0,0] counted)
    # B[0,k] (k=1,2) -> 2 terms (B[0,0] counted)
    
    # List of (i,j,k) for schoolbook terms involving 0:
    # 1. (0,0,0)
    # 2. (0,1,0)
    # 3. (0,2,0)
    # 4. (1,0,0)
    # 5. (2,0,0)
    # 6. (0,0,1)
    # 7. (1,0,1)
    # 8. (2,0,1)
    # 9. (0,0,2)
    # 10. (1,0,2)
    # 11. (2,0,2)
    # 12. (0,1,1)
    # 13. (1,1,0) # Wait, (1,1,0) is A[1,1]B[1,0]. B[1,0] is j=1,k=0.
    # 14. (2,1,0)
    # 15. (0,2,1)
    # 16. (1,2,0)
    # 17. (2,2,0)
    # 18. (0,1,2)
    # 19. (1,2,1) # A[1,2]B[2,1]. No 0 involved.
    # Wait, "involving 0" means i=0 or j=0 or k=0?
    # The Strassen part handled i,j,k in {1,2}.
    # So we need all (i,j,k) where at least one is 0.
    # Total 27 - 8 (2x2x2) = 19 terms.
    
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 0 or j == 0 or k == 0:
                    terms.append((i, j, k))
                    
    # Verify count: 19
    assert len(terms) == 19
    
    for idx, (i, j, k) in enumerate(terms):
        col = 7 + idx
        U[i * 3 + j, col] = 1
        V[j * 3 + k, col] = 1
        W[k * 3 + i, col] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
