# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    # We aim for the known optimal rank for <2, 2, 7> which is 25.
    # Structure: 3 blocks of <2,2,2> (rank 7) + 1 block of <2,2,1> (rank 4).
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen decomposition for 2x2
    # Indices: A: 0:A00, 1:A01, 2:A10, 3:A11
    # B: 0:B00, 1:B01, 2:B10, 3:B11
    # C: 0:C00, 1:C01, 2:C10, 3:C11 (local) -> Global W index mapping needed
    
    # Strassen steps (7 multiplications)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A part)
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Coefficients for V (B part)
    # Local indices 0:B00, 1:B01, 2:B10, 3:B11
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for W (C part)
    # Local indices 0:C00, 1:C01, 2:C10, 3:C11
    # C00 = M1 - M3 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M5 + M7
    W_strassen = [
        [1, 0, 0, 1],   # M1 contributes to C00, C11
        [0, 0, 1, 0],   # M2 contributes to C10
        [0, 1, 0, 0],   # M3 contributes to C01
        [0, 0, 1, 0],   # M4 contributes to C10 ... wait
        # Let's re-verify Strassen C reconstruction.
        # C00 = M1 - M3 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M5 + M7
        
        # So for each M, what is its coeff in C?
        # M1: C00:1, C11:1
        # M2: C10:1
        # M3: C01:1, C00:-1
        # M4: C10:1, C00:-1
        # M5: C01:1, C11:-1
        # M6: C00:1
        # M7: C11:1
    ]
    
    # Correct W_strassen matrix: rows are M1..M7, cols are C00, C01, C10, C11
    W_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [-1, 1, 0, 0], # M3
        [-1, 0, 1, 0], # M4
        [0, 1, 0, -1], # M5
        [1, 0, 0, 0],  # M6
        [0, 0, 0, 1]   # M7
    ]
    
    # Process 3 Strassen blocks (columns 0..5 of B)
    for b in range(3):
        for r in range(7):
            # Update U
            for i in range(4):
                U[i][col] += U_strassen[r][i]
            
            # Update V
            # B local 0,1,2,3 map to global B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
            # V index = row * p + col
            # Row 0: 0, 1
            # Row 1: 7, 8 (if p=7, 1*7+0=7, 1*7+1=8)
            v_indices = [0 * p + 2 * b, 0 * p + 2 * b + 1, 1 * p + 2 * b, 1 * p + 2 * b + 1]
            for j in range(4):
                V[v_indices[j]][col] += V_strassen[r][j]
                
            # Update W
            # C local 0,1,2,3 map to global C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1]
            # W index = col * n + row (Trace convention W[k*n+i, r] where k is col of C? 
            # Task says: W[k*n+i, r]. 
            # In matrix multiplication C = AB, C has size p x n.
            # Standard trace convention: W_{k,i} corresponds to C_{i,k}? 
            # "Output factor follows the official trace convention W[k*n+i,r]"
            # Usually, tensor T = sum_r U_r \otimes V_r \otimes W_r.
            # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
            # If C_{i,k} = sum_j A_{i,j} B_{j,k}.
            # Then T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}? No.
            # The tensor for matrix multiplication is usually defined such that 
            # sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
            # This is a contraction.
            # The decomposition T = sum U V W usually implies:
            # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) ... No.
            # Standard tensor decomposition of multiplication tensor:
            # The tensor T has indices (i, j, k) corresponding to A_{i,j}, B_{j,k}, C_{i,k}.
            # T_{i,j,k} = 1 if the term is present.
            # Decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}.
            # Here U is shape (n*m, rank), V is (m*p, rank), W is (p*n, rank).
            # U index: i*m + j.
            # V index: j*p + k.
            # W index: i*n + k.
            
            # Let's check the current program's W indexing.
            # Current: w_idx = (2 * b) * n + 0 for C00.
            # C00 is i=0, k=2b.
            # W index = i * n + k? 
            # If W index = i * n + k, then for i=0, k=2b, index = 2b.
            # Current code: w_idx = (2 * b) * n + 0. This is k * n + i.
            # So the current program uses W index = k * n + i.
            # The task says: "W[k*n+i,r]".
            # So k is the first index of W (row), i is the second (col)?
            # W shape is (p*n, rank).
            # So W[row, r] where row = k * n + i.
            # This matches k (col of C, row of B) and i (row of C, row of A).
            
            # So for C[i, k], W index is k * n + i.
            
            w_indices = [
                (2 * b) * n + 0,       # C[0, 2b] -> k=2b, i=0
                (2 * b + 1) * n + 0,   # C[0, 2b+1] -> k=2b+1, i=0
                (2 * b) * n + 1,       # C[1, 2b] -> k=2b, i=1
                (2 * b + 1) * n + 1    # C[1, 2b+1] -> k=2b+1, i=1
            ]
            for j in range(4):
                W[w_indices[j]][col] += W_strassen[r][j]
                
            col += 1
            
    # Schoolbook for last block (column 6 of B)
    # C[0, 6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1, 6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # 4 terms
    # S1: A00 B06 C06
    # S2: A01 B16 C06
    # S3: A10 B06 C16
    # S4: A11 B16 C16
    
    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    # V indices:
    # B[0,6] -> V[0*7+6] = 6
    # B[1,6] -> V[1*7+6] = 13
    schoolbook_V = [
        [1, 0], # B06
        [0, 1], # B16
        [1, 0], # B06
        [0, 1]  # B16
    ]
    
    # W indices:
    # C[0,6] -> W[6*2+0] = 12
    # C[1,6] -> W[6*2+1] = 13
    schoolbook_W = [
        [1, 0], # C06
        [1, 0], # C06
        [0, 1], # C16
        [0, 1]  # C16
    ]
    
    for r in range(4):
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]
            
        V[6][col] += schoolbook_V[r][0]
        V[13][col] += schoolbook_V[r][1]
        
        W[12][col] += schoolbook_W[r][0]
        W[13][col] += schoolbook_W[r][1]
        
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
