"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: m=1 (j=0)
    # Rank 20
    for i in range(n):
        for k in range(p):
            # U[i*m + 0]
            U[i * m + 0][col] = 1
            # V[0*p + k]
            V[0 * p + k][col] = 1
            # W[k*n + i]
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: m=2 (j=1, 2)
    # Split p=10 into 5 blocks of 2
    # Use Strassen for each 2x2x2 block
    # Strassen terms for 2x2 matrices A, B
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Local indices for A: (0,1), (0,2), (1,1), (1,2) -> j=1,2
    # Local indices for B: (1,k1), (1,k2), (2,k1), (2,k2) -> j=1,2, k=k1,k2
    # Local indices for C: (0,k1), (0,k2), (1,k1), (1,k2) -> i=0,1, k=k1,k2
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [A01, A02, A11, A12] (flattened i,j for j=1,2)
        # V: [B1k1, B1k2, B2k1, B2k2] (flattened j,k for j=1,2, k=k1,k2)
        # W: [C0k1, C0k2, C1k1, C1k2] (flattened k,i for k=k1,k2, i=0,1)
        
        # M1
        ({(0,1): 1, (1,2): 1}, {(1,0): 1, (2,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}), # W indices: (k,i)
        # M2
        ({(1,1): 1, (1,2): 1}, {(1,0): 1}, {(0,1): 1, (1,1): 1}),
        # M3
        ({(0,1): 1}, {(1,1): 1, (2,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,2): 1}, {(2,0): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,1): 1, (0,2): 1}, {(2,1): 1}, {(0,0): 1, (1,0): 1}),
        # M6
        ({(1,1): 1, (0,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}),
        # M7
        ({(0,2): 1, (1,2): -1}, {(2,0): 1, (2,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
    ]
    
    # Note: W coefficients in Strassen result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to add these to W.
    # Instead of defining W in the term, we define the contribution of each M_s to C.
    
    strassen_contributions = [
        # M1
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, # C00, C01, C10, C11? No.
        # C indices are (i, k).
        # C00 (i=0, k=0), C01 (i=0, k=1), C10 (i=1, k=0), C11 (i=1, k=1)
        # M1 contributes to C00, C01, C10, C11?
        # C11 (local 0,0) = M1...
        # C12 (local 0,1) = ...
        # C21 (local 1,0) = ...
        # C22 (local 1,1) = ...
        
        # Let's map local (r, c) to global (i, k_offset)
        # i=r, k=k1+c
    ]
    
    # Let's just hardcode the W updates for each Strassen term
    # Term 1: M1
    # C00 += M1, C01 += 0? No.
    # C_local[0,0] += M1
    # C_local[0,1] += 0
    # C_local[1,0] += 0
    # C_local[1,1] += M1
    
    # Wait, Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each block l:
    for l in range(5):
        k1 = 2 * l
        k2 = 2 * l + 1
        
        # M1
        U[(0*m+1)][col] = 1
        U[(1*m+2)][col] = 1
        V[(1*p+k1)][col] = 1
        V[(2*p+k2)][col] = 1
        W[(k1*n+0)][col] = 1
        W[(k2*n+1)][col] = 1
        col += 1
        
        # M2
        U[(1*m+1)][col] = 1
        U[(1*m+2)][col] = 1
        V[(1*p+k1)][col] = 1
        W[(k1*n+1)][col] = 1
        W[(k2*n+1)][col] = 1 # C22 gets -M2? No, C22 = M1 - M2 + ...
        # Wait, C21 (i=1, k=k1) gets M2. C22 (i=1, k=k2) gets -M2.
        # So W[(k1*n+1)] = 1, W[(k2*n+1)] = -1
        # Let's re-verify indices.
        # i=1, k=k1 -> C21.
        # i=1, k=k2 -> C22.
        # M2 contributes +1 to C21, -1 to C22.
        col -= 1 # Rollback
        U[(1*m+1)][col] = 1
        U[(1*m+2)][col] = 1
        V[(1*p+k1)][col] = 1
        W[(k1*n+1)][col] = 1
        W[(k2*n+1)][col] = -1
        col += 1
        
        # M3
        U[(0*m+1)][col] = 1
        V[(1*p+k2)][col] = 1
        V[(2*p+k2)][col] = -1
        W[(k2*n+0)][col] = 1
        W[(k2*n+1)][col] = 1 # C12 gets M3, C22 gets M3
        col += 1
        
        # M4
        U[(1*m+2)][col] = 1
        V[(2*p+k1)][col] = 1
        V[(1*p+k1)][col] = -1
        W[(k1*n+0)][col] = 1
        W[(k1*n+1)][col] = 1 # C11 gets M4, C21 gets M4
        col += 1
        
        # M5
        U[(0*m+1)][col] = 1
        U[(0*m+2)][col] = 1
        V[(2*p+k2)][col] = 1
        W[(k2*n+0)][col] = 1 # C12 gets M5
        W[(k1*n+0)][col] = -1 # C11 gets -M5
        col += 1
        
        # M6
        U[(1*m+1)][col] = 1
        U[(0*m+1)][col] = -1
        V[(1*p+k1)][col] = 1
        V[(1*p+k2)][col] = 1
        W[(k2*n+1)][col] = 1 # C22 gets M6
        col += 1
        
        # M7
        U[(0*m+2)][col] = 1
        U[(1*m+2)][col] = -1
        V[(2*p+k1)][col] = 1
        V[(2*p+k2)][col] = 1
        W[(k1*n+0)][col] = 1
        W[(k2*n+0)][col] = 1
        W[(k1*n+1)][col] = 1
        W[(k2*n+1)][col] = 1 # C11, C12, C21, C22 all get M7?
        # C11 gets M7. C22 gets M7.
        # C12, C21 do not get M7.
        # Correction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M7 only contributes to C11 and C22?
        # No, M7 is (A12-A22)(B21+B22).
        # In standard Strassen, M7 contributes to C11 and C22?
        # Let's check standard Strassen matrix:
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        # Yes.
        col -= 1
        U[(0*m+2)][col] = 1
        U[(1*m+2)][col] = -1
        V[(2*p+k1)][col] = 1
        V[(2*p+k2)][col] = 1
        W[(k1*n+0)][col] = 1
        W[(k2*n+1)][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
