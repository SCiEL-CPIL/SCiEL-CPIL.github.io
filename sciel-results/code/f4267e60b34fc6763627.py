import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # We will build lists of (index, coeff) for each rank-1 term
    # Then convert to matrices
    
    terms = []
    
    # Helper to add a term
    # u_idx: index in U (0..8)
    # v_idx: index in V (0..23)
    # w_idx: index in W (0..23)
    # coeff: coefficient
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        terms.append((u_idx, v_idx, w_idx, coeff))
        
    # 1. Strassen for 2x2x8 (4 groups of 2x2x2)
    # A rows 0,1; cols 0,1. B rows 0,1; cols 0..7. C rows 0,1; cols 0..7.
    # U indices for A[0:2, 0:2]:
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4 (using row-major i*m+j)
    # A[0,0] -> 0*3+0 = 0
    # A[0,1] -> 0*3+1 = 1
    # A[1,0] -> 1*3+0 = 3
    # A[1,1] -> 1*3+1 = 4
    
    # V indices for B[0:2, 0:2] (for a specific col group c0, c1)
    # B[0, c] -> 0*8+c = c
    # B[1, c] -> 1*8+c = 8+c
    
    # W indices for C[0:2, c0:c1]
    # C[0, c] -> c*3+0 = 3c
    # C[1, c] -> c*3+1 = 3c+1
    
    for g in range(4):
        c0 = g * 2
        c1 = g * 2 + 1
        
        # Strassen intermediates for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A22)(B12+B22)
        # M6 = (A12-A22)(B21+B22)
        # M7 = (A11-A12)(B11+B21)
        
        # We need to express these as linear combinations of U, V, W terms.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # (sum ua * Aa) * (sum vb * Bb) = sum ua*vb * Aa*Bb
        # So for each M, we add terms for each pair (Aa, Bb).
        
        # A indices:
        # A11 = A[0,0] = 0
        # A12 = A[0,1] = 1
        # A21 = A[1,0] = 3
        # A22 = A[1,1] = 4
        
        # B indices for this group:
        # B11 = B[0, c0] = c0
        # B12 = B[0, c1] = c1
        # B21 = B[1, c0] = 8+c0
        # B22 = B[1, c1] = 8+c1
        
        # W indices for this group:
        # C11 = C[0, c0] = 3*c0
        # C12 = C[0, c1] = 3*c1
        # C21 = C[1, c0] = 3*c0+1
        # C22 = C[1, c1] = 3*c1+1
        
        # M1
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        add_term(0, c0, 3*c0, 1)
        add_term(0, 8+c1, 3*c0, 1)
        add_term(4, c0, 3*c0, 1)
        add_term(4, 8+c1, 3*c0, 1)
        
        # M2
        # (A21+A22)B11 = A21B11 + A22B11
        add_term(3, c0, 3*c0, 1)
        add_term(4, c0, 3*c0, 1)
        
        # M3
        # A11(B21-B22) = A11B21 - A11B22
        add_term(0, 8+c0, 3*c0, 1)
        add_term(0, 8+c1, 3*c0, -1)
        
        # M4
        # A22(B12-B11) = A22B12 - A22B11
        add_term(4, c1, 3*c0, 1)
        add_term(4, c0, 3*c0, -1)
        
        # M5
        # (A11+A22)(B12+B22) = A11B12 + A11B22 + A22B12 + A22B22
        add_term(0, c1, 3*c0, 1)
        add_term(0, 8+c1, 3*c0, 1)
        add_term(4, c1, 3*c0, 1)
        add_term(4, 8+c1, 3*c0, 1)
        
        # M6
        # (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        add_term(1, 8+c0, 3*c0, 1)
        add_term(1, 8+c1, 3*c0, 1)
        add_term(4, 8+c0, 3*c0, -1)
        add_term(4, 8+c1, 3*c0, -1)
        
        # M7
        # (A11-A12)(B11+B21) = A11B11 + A11B21 - A12B11 - A12B21
        add_term(0, c0, 3*c0, 1)
        add_term(0, 8+c0, 3*c0, 1)
        add_term(1, c0, 3*c0, -1)
        add_term(1, 8+c0, 3*c0, -1)
        
        # Combine to C
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We added terms to W with index 3*c0 for all M's?
        # No, we need to add them to the correct C index with the correct sign.
        # My add_term calls above all used w_idx = 3*c0. That's wrong.
        # I need to accumulate the coefficients for each M, then add to C.
        
        # Let's restart the Strassen part with proper accumulation.
        terms.clear() # Clear previous incorrect adds
        
        # For each group g:
        
    # Re-do Strassen part properly
    for g in range(4):
        c0 = g * 2
        c1 = g * 2 + 1
        
        # Coefficients for each M
        # M1: (0, c0, 1), (0, 8+c1, 1), (4, c0, 1), (4, 8+c1, 1)
        # M2: (3, c0, 1), (4, c0, 1)
        # M3: (0, 8+c0, 1), (0, 8+c1, -1)
        # M4: (4, c1, 1), (4, c0, -1)
        # M5: (0, c1, 1), (0, 8+c1, 1), (4, c1, 1), (4, 8+c1, 1)
        # M6: (1, 8+c0, 1), (1, 8+c1, 1), (4, 8+c0, -1), (4, 8+c1, -1)
        # M7: (0, c0, 1), (0, 8+c0, 1), (1, c0, -1), (1, 8+c0, -1)
        
        # C11 (w_idx 3*c0) = M1 + M4 - M5 + M7
        # C12 (w_idx 3*c1) = M3 + M5
        # C21 (w_idx 3*c0+1) = M2 + M4
        # C22 (w_idx 3*c1+1) = M1 - M2 + M3 + M6
        
        # We will add terms directly to the final W index with the combined coefficient.
        
        # M1 terms:
        # (0, c0) -> C11 (+1), C22 (+1)
        # (0, 8+c1) -> C11 (+1), C22 (+1)
        # (4, c0) -> C11 (+1), C22 (+1)
        # (4, 8+c1) -> C11 (+1), C22 (+1)
        
        # M2 terms:
        # (3, c0) -> C21 (+1), C22 (-1)
        # (4, c0) -> C21 (+1), C22 (-1)
        
        # M3 terms:
        # (0, 8+c0) -> C12 (+1), C22 (+1)
        # (0, 8+c1) -> C12 (-1), C22 (+1)
        
        # M4 terms:
        # (4, c1) -> C11 (+1), C21 (+1)
        # (4, c0) -> C11 (-1), C21 (+1)
        
        # M5 terms:
        # (0, c1) -> C11 (-1), C12 (+1)
        # (0, 8+c1) -> C11 (-1), C12 (+1)
        # (4, c1) -> C11 (-1), C12 (+1)
        # (4, 8+c1) -> C11 (-1), C12 (+1)
        
        # M6 terms:
        # (1, 8+c0) -> C22 (+1)
        # (1, 8+c1) -> C22 (+1)
        # (4, 8+c0) -> C22 (-1)
        # (4, 8+c1) -> C22 (-1)
        
        # M7 terms:
        # (0, c0) -> C11 (+1)
        # (0, 8+c0) -> C11 (+1)
        # (1, c0) -> C11 (-1)
        # (1, 8+c0) -> C11 (-1)
        
        # Let's sum up coefficients for each (u_idx, v_idx) pair for each w_idx.
        
        # C11 (3*c0):
        # From M1: (0,c0)+1, (0,8+c1)+1, (4,c0)+1, (4,8+c1)+1
        # From M4: (4,c1)+1, (4,c0)-1
        # From M5: (0,c1)-1, (0,8+c1)-1, (4,c1)-1, (4,8+c1)-1
        # From M7: (0,c0)+1, (0,8+c0)+1, (1,c0)-1, (1,8+c0)-1
        
        # Combine:
        # (0,c0): 1+1 = 2
        # (0,8+c1): 1-1 = 0
        # (4,c0): 1-1 = 0
        # (4,8+c1): 1-1 = 0
        # (4,c1): 1-1 = 0
        # (0,c1): -1
        # (0,8+c0): 1
        # (1,c0): -1
        # (1,8+c0): -1
        
        add_term(0, c0, 3*c0, 2)
        add_term(0, c1, 3*c0, -1)
        add_term(0, 8+c0, 3*c0, 1)
        add_term(1, c0, 3*c0, -1)
        add_term(1, 8+c0, 3*c0, -1)
        
        # C12 (3*c1):
        # From M3: (0,8+c0)+1, (0,8+c1)-1
        # From M5: (0,c1)+1, (0,8+c1)+1, (4,c1)+1, (4,8+c1)+1
        
        # Combine:
        # (0,8+c0): 1
        # (0,8+c1): -1+1 = 0
        # (0,c1): 1
        # (4,c1): 1
        # (4,8+c1): 1
        
        add_term(0, 8+c0, 3*c1, 1)
        add_term(0, c1, 3*c1, 1)
        add_term(4, c1, 3*c1, 1)
        add_term(4, 8+c1, 3*c1, 1)
        
        # C21 (3*c0+1):
        # From M2: (3,c0)+1, (4,c0)+1
        # From M4: (4,c1)+1, (4,c0)+1
        
        # Combine:
        # (3,c0): 1
        # (4,c0): 1+1 = 2
        # (4,c1): 1
        
        add_term(3, c0, 3*c0+1, 1)
        add_term(4, c0, 3*c0+1, 2)
        add_term(4, c1, 3*c0+1, 1)
        
        # C22 (3*c1+1):
        # From M1: (0,c0)+1, (0,8+c1)+1, (4,c0)+1, (4,8+c1)+1
        # From M2: (3,c0)-1, (4,c0)-1
        # From M3: (0,8+c0)+1, (0,8+c1)+1
        # From M6: (1,8+c0)+1, (1,8+c1)+1, (4,8+c0)-1, (4,8+c1)-1
        
        # Combine:
        # (0,c0): 1
        # (0,8+c1): 1+1 = 2
        # (4,c0): 1-1 = 0
        # (4,8+c1): 1-1 = 0
        # (3,c0): -1
        # (0,8+c0): 1
        # (1,8+c0): 1
        # (1,8+c1): 1
        # (4,8+c0): -1
        
        add_term(0, c0, 3*c1+1, 1)
        add_term(0, 8+c1, 3*c1+1, 2)
        add_term(3, c0, 3*c1+1, -1)
        add_term(0, 8+c0, 3*c1+1, 1)
        add_term(1, 8+c0, 3*c1+1, 1)
        add_term(1, 8+c1, 3*c1+1, 1)
        add_term(4, 8+c0, 3*c1+1, -1)
        
    # 2. T2: 2x1x8
    # A rows 0,1; col 2. U indices: 0*3+2=2, 1*3+2=5.
    # B row 2; cols 0..7. V indices: 2*8+k = 16+k.
    # C rows 0,1; cols 0..7. W indices: k*3+0, k*3+1.
    for k in range(8):
        # C[0,k] = A[0,2] B[2,k]
        add_term(2, 16+k, k*3+0, 1)
        # C[1,k] = A[1,2] B[2,k]
        add_term(5, 16+k, k*3+1, 1)
        
    # 3. T3: 1x2x8
    # A row 2; cols 0,1. U indices: 2*3+0=6, 2*3+1=7.
    # B rows 0,1; cols 0..7. V indices: 0*8+k=k, 1*8+k=8+k.
    # C row 2; cols 0..7. W indices: k*3+2.
    for k in range(8):
        # C[2,k] = A[2,0] B[0,k] + A[2,1] B[1,k]
        add_term(6, k, k*3+2, 1)
        add_term(7, 8+k, k*3+2, 1)
        
    # 4. T4: 1x1x8
    # A row 2; col 2. U index: 2*3+2=8.
    # B row 2; cols 0..7. V indices: 16+k.
    # C row 2; cols 0..7. W indices: k*3+2.
    for k in range(8):
        # C[2,k] += A[2,2] B[2,k]
        add_term(8, 16+k, k*3+2, 1)
        
    # Build matrices
    rank = len(terms)
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    for r, (u, v, w, c) in enumerate(terms):
        U[u][r] = c
        V[v][r] = c
        W[w][r] = c
        
    return {"U": U, "V": V, "W": W}
