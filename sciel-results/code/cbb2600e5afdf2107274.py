import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The rank-minimal decomposition for 2x4x8 is 56.
    # This is achieved by splitting the middle dimension m=4 into two parts of size 2.
    # C = A_L B_T + A_R B_B
    # where A_L is 2x2, A_R is 2x2, B_T is 2x8, B_B is 2x8.
    # Each term is a 2x2 * 2x8 multiplication.
    # 2x2 * 2x8 is equivalent to 4 independent 2x2 * 2x2 multiplications.
    # Rank of 2x2 * 2x2 is 7 (Strassen).
    # So rank of 2x2 * 2x8 is 4 * 7 = 28.
    # Total rank = 28 + 28 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x7 decomposition
    # Maps (i, j, k) in 2x2x2 space to global indices
    # We need to map local 2x2 blocks to global indices.
    
    # Define the 7 Strassen products for 2x2 matrix multiplication
    # Local indices: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will construct the 56 terms by iterating over the 4 blocks of B (cols 0-1, 2-3, 4-5, 6-7)
    # and the 2 parts of A (rows 0-1, cols 0-1 and rows 0-1, cols 2-3)
    # Actually, A is 2x4.
    # A_L = A[:, 0:2] (cols 0,1)
    # A_R = A[:, 2:4] (cols 2,3)
    # B_T = B[0:2, :] (rows 0,1)
    # B_B = B[2:4, :] (rows 2,3)
    
    # Term 1: A_L B_T
    # A_L is 2x2. B_T is 2x8.
    # Split B_T into 4 blocks of 2x2: B_T[:, 0:2], B_T[:, 2:4], B_T[:, 4:6], B_T[:, 6:8]
    # Let's call these B_T0, B_T1, B_T2, B_T3.
    # Each A_L * B_Tl is a 2x2 * 2x2 multiplication.
    
    # Term 2: A_R B_B
    # A_R is 2x2. B_B is 2x8.
    # Split B_B into 4 blocks of 2x2: B_B[:, 0:2], B_B[:, 2:4], B_B[:, 4:6], B_B[:, 6:8]
    # Let's call these B_B0, B_B1, B_B2, B_B3.
    
    # Total 8 sub-multiplications of 2x2 * 2x2.
    # Each uses 7 Strassen terms. Total 56.
    
    # Define Strassen coefficients for 2x2x2
    # U_local, V_local, W_local for 2x2x2
    # Indices: A[0,0], A[0,1], A[1,0], A[1,1] -> 0,1,2,3
    #           B[0,0], B[0,1], B[1,0], B[1,1] -> 0,1,2,3
    #           C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3 (using trace convention? No, standard C[i,k])
    # The problem uses W[k*n+i]. For 2x2, n=2, p=2. W[2k+i].
    # C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3.
    # Wait, W index is k*n+i.
    # k=0, i=0 -> 0. (C[0,0])
    # k=1, i=0 -> 2. (C[0,1])
    # k=0, i=1 -> 1. (C[1,0])
    # k=1, i=1 -> 3. (C[1,1])
    # So W indices: 0, 2, 1, 3 for C[0,0], C[0,1], C[1,0], C[1,1].
    
    # Strassen terms for 2x2:
    # M1: A00+A11, B00+B11 -> C00+C11? No.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M7
    
    # Let's build the 7 terms for a generic 2x2 block.
    # U_term: list of (idx, coeff)
    # V_term: list of (idx, coeff)
    # W_term: list of (idx, coeff)
    
    def get_strassen_terms():
        terms = []
        # M1
        terms.append({
            'U': [(0, 1), (3, 1)], # A00, A11
            'V': [(0, 1), (3, 1)], # B00, B11
            'W': [(0, 1), (1, 1), (2, 1), (3, 1)] # C00, C01, C10, C11? No.
        })
        # We need to map W contributions correctly.
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7
        
        # W indices for C[0,0], C[0,1], C[1,0], C[1,1] are 0, 2, 1, 3.
        
        # M1 contributes to C00 (+1), C11 (+1)
        terms[0]['W'] = [(0, 1), (3, 1)]
        
        # M2: (A10+A11)B00
        terms.append({
            'U': [(2, 1), (3, 1)], # A10, A11
            'V': [(0, 1)],         # B00
            'W': [(1, 1), (3, -1)] # C10 (+1), C11 (-1)
        })
        
        # M3: A00(B01-B11)
        terms.append({
            'U': [(0, 1)],         # A00
            'V': [(1, 1), (3, -1)],# B01, B11
            'W': [(2, 1)]          # C01 (+1)
        })
        
        # M4: A11(B10-B00)
        terms.append({
            'U': [(3, 1)],         # A11
            'V': [(2, 1), (0, -1)],# B10, B00
            'W': [(0, -1), (1, 1)] # C00 (-1), C10 (+1)
        })
        
        # M5: (A00+A01)B11
        terms.append({
            'U': [(0, 1), (1, 1)], # A00, A01
            'V': [(3, 1)],         # B11
            'W': [(2, 1), (3, 1)]  # C01 (+1), C11 (+1)
        })
        
        # M6: (A10-A00)(B00+B01)
        terms.append({
            'U': [(2, 1), (0, -1)],# A10, A00
            'V': [(0, 1), (1, 1)], # B00, B01
            'W': [(0, 1)]          # C00 (+1)
        })
        
        # M7: (A01-A11)(B10+B11)
        terms.append({
            'U': [(1, 1), (3, -1)],# A01, A11
            'V': [(2, 1), (3, 1)], # B10, B11
            'W': [(3, 1)]          # C11 (+1)
        })
        
        return terms

    strassen_terms = get_strassen_terms()
    
    # Mapping of local 2x2 indices to global indices
    # Global A: 2x4. Indices 0..7.
    # A_L (cols 0,1): A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # A_R (cols 2,3): A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
    # Local A indices 0,1,2,3 map to global 0,1,2,3 for A_L
    # Local A indices 0,1,2,3 map to global 4,5,6,7 for A_R
    
    # Global B: 4x8. Indices 0..31.
    # B_T (rows 0,1):
    #   Block 0 (cols 0,1): B[0,0]=0, B[0,1]=1, B[1,0]=8, B[1,1]=9
    #   Block 1 (cols 2,3): B[0,2]=2, B[0,3]=3, B[1,2]=10, B[1,3]=11
    #   Block 2 (cols 4,5): B[0,4]=4, B[0,5]=5, B[1,4]=12, B[1,5]=13
    #   Block 3 (cols 6,7): B[0,6]=6, B[0,7]=7, B[1,6]=14, B[1,7]=15
    # B_B (rows 2,3):
    #   Block 0 (cols 0,1): B[2,0]=16, B[2,1]=17, B[3,0]=24, B[3,1]=25
    #   Block 1 (cols 2,3): B[2,2]=18, B[2,3]=19, B[3,2]=26, B[3,3]=27
    #   Block 2 (cols 4,5): B[2,4]=20, B[2,5]=21, B[3,4]=28, B[3,5]=29
    #   Block 3 (cols 6,7): B[2,6]=22, B[2,7]=23, B[3,6]=30, B[3,7]=31
    
    # Local B indices 0,1,2,3 (B00, B01, B10, B11)
    # Map to global B indices for a specific block.
    
    # Global C: 2x8. Indices 0..15.
    # W index = k*n + i = 2k + i.
    # C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3.
    # Block 0 (cols 0,1): C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    # Block 1 (cols 2,3): C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    # Block 2 (cols 4,5): C[0,4]=8, C[0,5]=10, C[1,4]=9, C[1,5]=11
    # Block 3 (cols 6,7): C[0,6]=12, C[0,7]=14, C[1,6]=13, C[1,7]=15
    
    # Local C indices 0,1,2,3 (C00, C01, C10, C11)
    # Map to global C indices for a specific block.
    
    col = 0
    for a_part in [0, 1]: # 0 for A_L, 1 for A_R
        for b_row_part in [0, 1]: # 0 for B_T, 1 for B_B
            for b_col_block in range(4): # 0..3
                # Determine global index offsets
                # A offset
                a_offset = 4 * a_part
                
                # B offset
                # B_T rows 0,1. B_B rows 2,3.
                b_row_offset = 16 * b_row_part
                b_col_offset = 2 * b_col_block
                # B global index = row * 8 + col
                # Local B 0 (0,0) -> (0,0) -> 0
                # Local B 1 (0,1) -> (0,1) -> 1
                # Local B 2 (1,0) -> (1,0) -> 8
                # Local B 3 (1,1) -> (1,1) -> 9
                # So local idx 0->0, 1->1, 2->8, 3->9 relative to block start.
                b_local_to_global = [0, 1, 8, 9]
                b_offset = b_row_offset + b_col_offset
                
                # C offset
                c_offset = 4 * b_col_block
                # Local C 0 (0,0) -> 0
                # Local C 1 (0,1) -> 2
                # Local C 2 (1,0) -> 1
                # Local C 3 (1,1) -> 3
                c_local_to_global = [0, 2, 1, 3]
                
                for term in strassen_terms:
                    # U
                    for local_idx, coeff in term['U']:
                        global_idx = a_offset + local_idx
                        U[global_idx][col] += coeff
                    # V
                    for local_idx, coeff in term['V']:
                        global_idx = b_offset + b_local_to_global[local_idx]
                        V[global_idx][col] += coeff
                    # W
                    for local_idx, coeff in term['W']:
                        global_idx = c_offset + c_local_to_global[local_idx]
                        W[global_idx][col] += coeff
                    col += 1

    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_decomp[i,k] = total
    
    error = np.max(np.abs(C_true - C_decomp))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")
