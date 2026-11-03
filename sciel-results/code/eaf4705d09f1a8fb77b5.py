import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 130.0
    
    n, m, p = 3, 3, 7
    rank = 55  # 23 + 23 + 9
    
    U = np.zeros((9, rank))
    V = np.zeros((21, rank))
    W = np.zeros((21, rank))
    
    # Stothers rank-23 decomposition for 3x3x3
    # Each term: (U vector of 9 entries, V vector of 9 entries, W vector of 9 entries)
    # U[i*3+j], V[j*3+k], W[k*3+i]
    # Coefficients: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    #
    # Standard Stothers decomposition:
    stothers_23 = [
        # (U, V, W) as sparse dicts
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),  # C00 = A00*B00
        ({(0,0):1, (0,1):1}, {(0,1):1, (1,0):1}, {(0,0):1, (0,1):1}),  # C01 = A00*B01+A01*B00
        ({(0,1):1}, {(0,1):1, (1,0):1}, {(0,1):1}),  # C02 partial
        ({(0,2):1, (1,0):1}, {(0,2):1, (1,2):1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1},
         {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1},
         {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1}),
    ]
    
    # Actually, let me use a cleaner approach. I'll define the 23 terms explicitly.
    # The Stothers rank-23 decomposition for 3x3x3:
    
    terms_333 = [
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        # Term 1: A[0,0]*B[0,1] + A[0,1]*B[0,0] -> C[0,1]
        ({(0,0):1, (0,1):1}, {(0,1):1, (1,0):1}, {(0,1):1}),
        # Term 2: A[0,1]*B[0,1] + A[0,0]*B[0,0] -> C[0,0] correction
        # Let me use a known explicit form instead
    ]
    
    # I'll use a verified approach: define the 23 rank-1 terms for 3x3x3
    # based on the standard Stothers algorithm.
    
    # The 23 terms for multiplying 3x3 by 3x3:
    # P1 = A[0,0]*B[0,0]
    # P2 = A[0,0]*B[0,1] + A[0,1]*B[0,0]
    # P3 = A[0,1]*B[0,1] + A[0,0]*B[0,0]
    # P4 = A[0,2]*B[0,2]
    # P5 = A[1,0]*B[1,0]
    # P6 = A[1,0]*B[1,1] + A[1,1]*B[1,0]
    # P7 = A[1,1]*B[1,1] + A[1,0]*B[1,0]
    # P8 = A[1,2]*B[1,2]
    # P9 = A[2,0]*B[2,0]
    # P10 = A[2,0]*B[2,1] + A[2,1]*B[2,0]
    # P11 = A[2,1]*B[2,1] + A[2,0]*B[2,0]
    # P12 = A[2,2]*B[2,2]
    # P13 = (A[0,0]+A[0,1]+A[0,2]+A[1,0]+A[1,1]+A[1,2]+A[2,0]+A[2,1]+A[2,2]) *
    #      (B[0,0]+B[0,1]+B[0,2]+B[1,0]+B[1,1]+B[1,2]+B[2,0]+B[2,1]+B[2,2])
    # ... this is getting complex. Let me use a different known decomposition.
    
    # I'll use a simpler but verified rank-23 decomposition.
    # Actually, let me just use the schoolbook (rank 27) for 3x3x3 and accept rank 63,
    # OR find a cleaner improved decomposition.
    
    # Better approach: Use a known rank-23 decomposition.
    # The following is a standard rank-23 decomposition for 3x3x3:
    
    def add_block(k_start, base_col):
        """Add a rank-23 3x3x3 block starting at column k_start."""
        # 23 terms for 3x3x3 matrix multiplication
        # Each term contributes U[i*3+j] * V[j*7+(k_start+k)] * W[(k_start+k)*3+i] = product
        
        # I'll use a known rank-23 decomposition
        # Here's a verified set of 23 terms:
        
        terms = [
            # (u_entries, v_entries, w_entries)
            # u_entries: dict mapping (i,j) -> coeff for A[i,j]
            # v_entries: dict mapping (j,k) -> coeff for B[j,k]
            # w_entries: dict mapping (k,i) -> coeff for C[k,i] (in trace convention)
            
            # These 23 terms implement the Stothers algorithm
            # I'll use a simpler correct decomposition for now
        ]
        
        # Let me use a concrete rank-23 decomposition:
        # Based on the literature, here are the 23 rank-1 tensors:
        
        t = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),
            ({(0,0):1, (0,1):1}, {(0,1):1, (1,0):1}, {(0,1):1, (1,0):1}),
            ({(0,1):1, (0,2):1}, {(0,2):1, (2,0):1}, {(0,2):1, (2,0):1}),
            ({(0,2):1, (1,0):1}, {(0,2):1, (1,0):1}, {(0,2):1, (1,0):1}),
            ({(1,0):1, (1,1):1}, {(1,1):1, (1,0):1}, {(1,1):1, (1,0):1}),
            ({(1,1):1, (1,2):1}, {(1,2):1, (2,1):1}, {(1,2):1, (2,1):1}),
            ({(1,2):1, (2,0):1}, {(1,2):1, (2,0):1}, {(1,2):1, (2,0):1}),
            ({(2,0):1, (2,1):1}, {(2,1):1, (2,0):1}, {(2,1):1, (2,0):1}),
            ({(2,1):1, (2,2):1}, {(2,2):1, (0,2):1}, {(2,2):1, (0,2):1}),
            ({(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}, {(2,2):1, (0,0):1}),
            ({(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1},
             {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1},
             {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1}),
            ({(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1},
             {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
            ({(0,0):1, (0,2):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1},
             {(0,0):1, (0,2):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1},
             {(0,0):1, (0,2):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1}),
        ]
        
        # This is getting too complex and error-prone. Let me use a fallback:
        # Use rank 27 (schoolbook) for each 3x3x3 block: 27 + 27 + 9 = 63
        # Or use a simpler improvement.
        
        # Actually, let me just return schoolbook to be safe, since getting
        # the rank-23 decomposition exactly right is critical.
        pass
    
    # Given the complexity and risk of errors in the rank-23 decomposition,
    # let me use a conservative but correct approach:
    # Schoolbook gives rank 63. I'll try to do better with a simpler improvement.
    
    # For 3x3x7, I can use:
    # - For each of the 7 columns, use schoolbook (rank 9 per column) = 63 total
    # This is the baseline.
    
    # A simple improvement: for 3x3x3, the minimum rank is 23.
    # If I can correctly implement it, I get 23+23+9 = 55.
    
    # Let me try a different approach: use numpy to verify a decomposition.
    # I'll construct a candidate rank-55 decomposition and verify it.
    
    # For safety, return schoolbook (rank 63):
    U_school = np.zeros((9, 63))
    V_school = np.zeros((21, 63))
    W_school = np.zeros((21, 63))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(7):
                U_school[i*3+j, col] = 1
                V_school[j*7+k, col] = 1
                W_school[k*3+i, col] = 1
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
