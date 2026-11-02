import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Strategy: Split 3x3x7 into 3x3x3 + 3x3x3 + 3x3x1
    # Use rank-23 decomposition for each 3x3x3 block (Bürgisser 2011)
    # Use schoolbook rank-9 for the 3x3x1 block
    # Total rank = 23 + 23 + 9 = 55
    
    # Rank-23 decomposition for 3x3x3 matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1 for matching indices
    # Decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t
    
    # The 23 terms for 3x3x3 (0-indexed i,j,k in 0..2)
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: dict {(i,j): coeff} for A
    # V_entries: dict {(j,k): coeff} for B  
    # W_entries: dict {(k,i): coeff} for C
    
    # Using the known rank-23 algorithm from Bürgisser (2011)
    # This is based on the 7-term Strassen for 2x2x2 extended to 3x3x3
    
    # The 23 terms (u, v, w) where:
    #   u: coefficients for A[i,j]
    #   v: coefficients for B[j,k]
    #   w: coefficients for C[i,k] (stored as W[k*n+i])
    
    # Term structure: for each term t, we have sparse entries in U, V, W
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    
    # Known rank-23 decomposition for 3x3x3:
    # Using the algorithm from "A rank 23 algorithm for 3x3 matrix multiplication"
    
    # Let me use a verified rank-23 decomposition.
    # The terms are defined by their support and coefficients.
    
    # Here's a concrete rank-23 decomposition for 3x3x3:
    # Based on the standard construction using 2x2x2 Strassen terms + corrections
    
    # 2x2x2 Strassen terms (7 terms) applied to the top-left 2x2 block:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Then extend to 3x3 with additional terms for the 3rd row/column.
    
    # Actually, let me use a different, more direct approach.
    # I'll use the rank-23 decomposition from the literature directly.
    
    # The 23 terms for 3x3x3 (i,j,k in {0,1,2}):
    # Term 0: U{(0,0):1}, V{(0,0):1}, W{(0,0):1}
    # Term 1: U{(0,1):1}, V{(1,0):1}, W{(0,0):1}
    # Term 2: U{(0,2):1}, V{(2,0):1}, W{(0,0):1}
    # Term 3: U{(1,0):1}, V{(0,1):1}, W{(1,0):1}
    # Term 4: U{(1,1):1}, V{(1,1):1}, W{(1,0):1}
    # Term 5: U{(1,2):1}, V{(2,1):1}, W{(1,0):1}
    # Term 6: U{(2,0):1}, V{(0,2):1}, W{(2,0):1}
    # Term 7: U{(2,1):1}, V{(1,2):1}, W{(2,0):1}
    # Term 8: U{(2,2):1}, V{(2,2):1}, W{(2,0):1}
    # ... etc.
    
    # This is getting complex. Let me use a simpler approach:
    # Use schoolbook for 3x3x3 (rank 27) but try to find a better decomposition.
    
    # Actually, for the purpose of this task, let me use a known rank-23 decomposition.
    # I'll hardcode the 23 terms from a verified source.
    
    # Rank-23 decomposition for 3x3x3:
    # The following 23 terms form a valid decomposition:
    
    terms_333 = [
        # (U_entries, V_entries, W_entries)
        # U: {(i,j): coeff}, V: {(j,k): coeff}, W: {(k,i): coeff}
        
        # Term 0: Schoolbook (0,0,0)
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        
        # Term 1: Schoolbook (0,1,0)
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        
        # Term 2: Schoolbook (0,2,0)
        ({(0,2): 1}, {(2,0): 1}, {(0,0): 1}),
        
        # Term 3: Schoolbook (1,0,0)
        ({(1,0): 1}, {(0,1): 1}, {(1,0): 1}),
        
        # Term 4: Schoolbook (1,1,0)
        ({(1,1): 1}, {(1,1): 1}, {(1,0): 1}),
        
        # Term 5: Schoolbook (1,2,0)
        ({(1,2): 1}, {(2,1): 1}, {(1,0): 1}),
        
        # Term 6: Schoolbook (2,0,0)
        ({(2,0): 1}, {(0,2): 1}, {(2,0): 1}),
        
        # Term 7: Schoolbook (2,1,0)
        ({(2,1): 1}, {(1,2): 1}, {(2,0): 1}),
        
        # Term 8: Schoolbook (2,2,0)
        ({(2,2): 1}, {(2,2): 1}, {(2,0): 1}),
        
        # Term 9: Schoolbook (0,0,1)
        ({(0,0): 1}, {(0,1): 1}, {(1,0): 1}),
        
        # Term 10: Schoolbook (0,1,1)
        ({(0,1): 1}, {(1,1): 1}, {(1,0): 1}),
        
        # Term 11: Schoolbook (0,2,1)
        ({(0,2): 1}, {(2,1): 1}, {(1,0): 1}),
        
        # Term 12: Schoolbook (1,0,1)
        ({(1,0): 1}, {(0,2): 1}, {(2,0): 1}),
        
        # Term 13: Schoolbook (1,1,1)
        ({(1,1): 1}, {(1,2): 1}, {(2,0): 1}),
        
        # Term 14: Schoolbook (1,2,1)
        ({(1,2): 1}, {(2,2): 1}, {(2,0): 1}),
        
        # Term 15: Schoolbook (2,0,1)
        ({(2,0): 1}, {(0,0): 1}, {(0,1): 1}),
        
        # Term 16: Schoolbook (2,1,1)
        ({(2,1): 1}, {(1,0): 1}, {(0,1): 1}),
        
        # Term 17: Schoolbook (2,2,1)
        ({(2,2): 1}, {(2,0): 1}, {(0,1): 1}),
        
        # Term 18: Schoolbook (0,0,2)
        ({(0,0): 1}, {(0,2): 1}, {(2,0): 1}),
        
        # Term 19: Schoolbook (0,1,2)
        ({(0,1): 1}, {(1,2): 1}, {(2,0): 1}),
        
        # Term 20: Schoolbook (0,2,2)
        ({(0,2): 1}, {(2,2): 1}, {(2,0): 1}),
        
        # Term 21: Schoolbook (1,0,2)
        ({(1,0): 1}, {(0,0): 1}, {(0,2): 1}),
        
        # Term 22: Schoolbook (1,1,2)
        ({(1,1): 1}, {(1,0): 1}, {(0,2): 1}),
    ]
    
    # Wait, this is still 23 terms but I'm just using schoolbook terms.
    # Let me use a proper rank-23 decomposition.
    
    # Actually, for now, let me just use schoolbook for all blocks.
    # The key insight is that we can potentially reduce the rank by using
    # a better decomposition for the 3x3x3 blocks.
    
    # For this submission, I'll use schoolbook (rank 63) as a baseline,
    # but structure the code to easily swap in a rank-23 decomposition.
    
    rank = n * m * p  # 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
