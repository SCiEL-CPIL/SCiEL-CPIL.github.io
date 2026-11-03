"""Improved program: optimized block-Strassen for 4x4 * 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)  
    # W: (p*n, rank) = (24, rank)
    
    # Use block-Strassen decomposition
    # Split into 2x2 blocks:
    # A (4x4) -> 4 blocks of 2x2: A00, A01, A10, A11
    # B (4x6) -> 6 blocks of 2x2: B00, B01, B02, B10, B11, B12
    # C (4x6) -> 6 blocks of 2x2: C00, C01, C02, C10, C11, C12
    #
    # C[ci,cj] = A[ci,0]*B[0,cj] + A[ci,1]*B[1,cj]
    # Each block product sum needs rank 14 (two 2x2 Strassen)
    # Total: 6 * 14 = 84
    
    # Strassen 2x2 decomposition
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 2x2 Strassen factors
    # U_sub[i*2+j, r], V_sub[j*2+k, r], W_sub[k*2+i, r]
    U_sub = np.zeros((4, 7))
    V_sub = np.zeros((4, 7))
    W_sub = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    U_sub[0,0] = 1; U_sub[3,0] = 1  # A00, A11
    V_sub[0,0] = 1; V_sub[3,0] = 1  # B00, B11
    W_sub[0,0] = 1; W_sub[3,0] = 1  # C00, C11
    
    # M2 = (A10+A11)*B00
    U_sub[2,1] = 1; U_sub[3,1] = 1  # A10, A11
    V_sub[0,1] = 1                   # B00
    W_sub[2,1] = 1; W_sub[3,1] = -1  # C10, -C11
    
    # M3 = A00*(B01-B11)
    U_sub[0,2] = 1                   # A00
    V_sub[1,2] = 1; V_sub[3,2] = -1  # B01, -B11
    W_sub[1,2] = 1; W_sub[3,2] = 1   # C01, C11
    
    # M4 = A11*(B10-B00)
    U_sub[3,3] = 1                   # A11
    V_sub[2,3] = 1; V_sub[0,3] = -1  # B10, -B00
    W_sub[0,3] = 1; W_sub[2,3] = 1   # C00, C10
    
    # M5 = (A00+A01)*B11
    U_sub[0,4] = 1; U_sub[1,4] = 1   # A00, A01
    V_sub[3,4] = 1                   # B11
    W_sub[0,4] = -1; W_sub[1,4] = 1  # -C00, C01
    
    # M6 = (A10-A00)*(B00+B01)
    U_sub[2,5] = 1; U_sub[0,5] = -1  # A10, -A00
    V_sub[0,5] = 1; V_sub[1,5] = 1   # B00, B01
    W_sub[3,5] = 1                   # C11
    
    # M7 = (A01-A11)*(B10+B11)
    U_sub[1,6] = 1; U_sub[3,6] = -1  # A01, -A11
    V_sub[2,6] = 1; V_sub[3,6] = 1   # B10, B11
    W_sub[0,6] = 1; W_sub[1,6] = 1   # C00, C01
    
    # Now build the full decomposition
    # For each output block (ci, cj) with ci in {0,1}, cj in {0,1,2}:
    # C[ci,cj] = A[ci,0]*B[0,cj] + A[ci,1]*B[1,cj]
    # Use two Strassen decompositions: one for A[ci,0]*B[0,cj], one for A[ci,1]*B[1,cj]
    
    total_rank = 6 * 14  # 6 blocks, 14 rank each
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    for ci in range(2):
        for cj in range(3):
            base = (ci * 3 + cj) * 14
            
            # First Strassen: A[ci,0] * B[0,cj]
            # A[ci,0]: rows 2*ci to 2*ci+1, cols 0-1
            # B[0,cj]: rows 0-1, cols 2*cj to 2*cj+1
            # C[ci,cj]: rows 2*ci to 2*ci+1, cols 2*cj to 2*cj+1
            
            for r in range(7):
                col = base + r
                # U: A[ci,0] elements
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + lj
                        U[a_idx, col] += U_sub[li*2+lj, r]
                # V: B[0,cj] elements
                for li in range(2):
                    for lj in range(2):
                        b_idx = li * 6 + (2*cj + lj)
                        V[b_idx, col] += V_sub[li*2+lj, r]
                # W: C[ci,cj] elements
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (2*cj + lj)
                        W[c_idx, col] += W_sub[li*2+lj, r]
            
            # Second Strassen: A[ci,1] * B[1,cj]
            # A[ci,1]: rows 2*ci to 2*ci+1, cols 2-3
            # B[1,cj]: rows 2-3, cols 2*cj to 2*cj+1
            
            for r in range(7):
                col = base + 7 + r
                # U: A[ci,1] elements
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + (2 + lj)
                        U[a_idx, col] += U_sub[li*2+lj, r]
                # V: B[1,cj] elements
                for li in range(2):
                    for lj in range(2):
                        b_idx = (2 + li) * 6 + (2*cj + lj)
                        V[b_idx, col] += V_sub[li*2+lj, r]
                # W: C[ci,cj] elements (same as first)
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (2*cj + lj)
                        W[c_idx, col] += W_sub[li*2+lj, r]
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
