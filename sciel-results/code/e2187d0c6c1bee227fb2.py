import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 5 blocks of 2x2x2 (Strassen, rank 7) + 5 blocks of 2x1x2 (Schoolbook, rank 4)
    # Total rank = 5*7 + 5*4 = 35 + 20 = 55
    rank = 55
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 0
    
    # Part 1: j in {0, 1} (2x2 part of A and B)
    # A indices: i in {0,1}, j in {0,1} -> row i*m + j
    # B indices: j in {0,1}, k in {0..9} -> row j*p + k
    # C indices: i in {0,1}, k in {0..9} -> row k*n + i
    
    for block_idx in range(5):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B01)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to the global indices.
        # A00 is A[0,0] -> U row 0*m+0 = 0
        # A11 is A[1,1] -> U row 1*m+1 = 4
        # A10 is A[1,0] -> U row 1*m+0 = 3
        # A01 is A[0,1] -> U row 0*m+1 = 1
        
        # B00 is B[0,k0] -> V row 0*p+k0
        # B11 is B[1,k1] -> V row 1*p+k1
        # B01 is B[0,k1] -> V row 0*p+k1
        # B10 is B[1,k0] -> V row 1*p+k0
        
        # C00 is C[0,k0] -> W row k0*n+0
        # C11 is C[1,k1] -> W row k1*n+1
        # C01 is C[0,k1] -> W row k1*n+0
        # C10 is C[1,k0] -> W row k0*n+1
        
        # Define the 7 terms
        # Each term is (U_entries, V_entries, W_coeff)
        # U_entries: dict {row: val}
        # V_entries: dict {row: val}
        # W_coeff: dict {row: val}
        
        terms = []
        
        # M1
        terms.append((
            {0: 1, 4: 1}, 
            {0*p+k0: 1, 1*p+k1: 1}, 
            {k0*n+0: 1, k1*n+1: 1, k1*n+0: -1, k0*n+1: 1} 
        ))
        
        # M2
        terms.append((
            {3: 1, 4: 1}, 
            {0*p+k0: 1}, 
            {k0*n+1: 1, k1*n+1: -1} 
        ))
        
        # M3
        terms.append((
            {0: 1}, 
            {1*p+k0: 1, 1*p+k1: -1}, 
            {k1*n+0: 1, k0*n+1: 1} 
        ))
        
        # M4
        terms.append((
            {4: 1}, 
            {0*p+k0: 1, 0*p+k1: -1}, 
            {k0*n+0: 1, k0*n+1: 1} 
        ))
        
        # M5
        terms.append((
            {0: 1, 3: 1}, 
            {1*p+k1: 1}, 
            {k0*n+0: -1, k1*n+0: 1} 
        ))
        
        # M6
        terms.append((
            {3: 1, 0: -1}, 
            {0*p+k0: 1, 0*p+k1: 1}, 
            {k1*n+1: 1} 
        ))
        
        # M7
        terms.append((
            {1: 1, 4: -1}, 
            {1*p+k0: 1, 1*p+k1: 1}, 
            {k0*n+0: 1, k1*n+1: 1} 
        ))
        
        for t in terms:
            u_dict, v_dict, w_dict = t
            for r, val in u_dict.items():
                U[r, col_offset] += val
            for r, val in v_dict.items():
                V[r, col_offset] += val
            for r, val in w_dict.items():
                W[r, col_offset] += val
            col_offset += 1
            
    # Part 2: j=2 (2x1 part of A and B)
    # A indices: i in {0,1}, j=2 -> row i*m+2
    # B indices: j=2, k in {0..9} -> row 2*p+k
    # C indices: i in {0,1}, k in {0..9} -> row k*n+i
    
    for block_idx in range(5):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # Schoolbook for 2x1x2
        # C00 = A02 B2k0
        # C01 = A02 B2k1
        # C10 = A12 B2k0
        # C11 = A12 B2k1
        
        # A02 is A[0,2] -> U row 0*m+2 = 2
        # A12 is A[1,2] -> U row 1*m+2 = 5
        # B2k0 is B[2,k0] -> V row 2*p+k0
        # B2k1 is B[2,k1] -> V row 2*p+k1
        # C00 is C[0,k0] -> W row k0*n+0
        # C01 is C[0,k1] -> W row k1*n+0
        # C10 is C[1,k0] -> W row k0*n+1
        # C11 is C[1,k1] -> W row k1*n+1
        
        # Term 1: A02 B2k0 -> C00
        U[2, col_offset] = 1
        V[2*p+k0, col_offset] = 1
        W[k0*n+0, col_offset] = 1
        col_offset += 1
        
        # Term 2: A02 B2k1 -> C01
        U[2, col_offset] = 1
        V[2*p+k1, col_offset] = 1
        W[k1*n+0, col_offset] = 1
        col_offset += 1
        
        # Term 3: A12 B2k0 -> C10
        U[5, col_offset] = 1
        V[2*p+k0, col_offset] = 1
        W[k0*n+1, col_offset] = 1
        col_offset += 1
        
        # Term 4: A12 B2k1 -> C11
        U[5, col_offset] = 1
        V[2*p+k1, col_offset] = 1
        W[k1*n+1, col_offset] = 1
        col_offset += 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
