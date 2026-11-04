"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank decomposition for 3x3 x 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Helper to create a rank-1 term for a specific (i,j,k) triple
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def make_term(i, j, k, n, m, p):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        u[i*m + j] = 1
        v[j*p + k] = 1
        w[k*n + i] = 1
        return u, v, w

    # Strassen 2x2x2 decomposition
    # Indices for 2x2 block: i, j, k in {0, 1}
    # We need to map these to the global indices.
    # This function returns a list of (u, v, w) tuples for the 2x2x2 block
    # assuming it is located at offset (i0, j0, k0) in the larger matrix.
    def strassen_2x2(i0, j0, k0, n, m, p):
        terms = []
        # M1 = (A11+A22)(B11+B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+0)*m + (j0+0)] = 1; u[(i0+1)*m + (j0+1)] = 1
        v[(j0+0)*p + (k0+0)] = 1; v[(j0+1)*p + (k0+1)] = 1
        w[(k0+0)*n + (i0+0)] = 1; w[(k0+1)*n + (i0+1)] = 1
        terms.append((u, v, w))
        
        # M2 = (A21+A22)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m + (j0+0)] = 1; u[(i0+1)*m + (j0+1)] = 1
        v[(j0+0)*p + (k0+0)] = 1
        w[(k0+0)*n + (i0+1)] = 1
        terms.append((u, v, w))
        
        # M3 = A11(B21-B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+0)*m + (j0+0)] = 1
        v[(j0+1)*p + (k0+0)] = 1; v[(j0+1)*p + (k0+1)] = -1
        w[(k0+0)*n + (i0+0)] = 1; w[(k0+1)*n + (i0+0)] = 1
        terms.append((u, v, w))
        
        # M4 = A22(B12-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m + (j0+1)] = 1
        v[(j0+0)*p + (k0+1)] = 1; v[(j0+0)*p + (k0+0)] = -1
        w[(k0+1)*n + (i0+1)] = 1; w[(k0+0)*n + (i0+1)] = 1
        terms.append((u, v, w))
        
        # M5 = (A11+A12)B22
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+0)*m + (j0+0)] = 1; u[(i0+0)*m + (j0+1)] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[(k0+1)*n + (i0+0)] = 1
        terms.append((u, v, w))
        
        # M6 = (A21-A11)(B11+B12)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m + (j0+0)] = 1; u[(i0+0)*m + (j0+0)] = -1
        v[(j0+0)*p + (k0+0)] = 1; v[(j0+0)*p + (k0+1)] = 1
        w[(k0+0)*n + (i0+1)] = 1; w[(k0+1)*n + (i0+1)] = 1
        terms.append((u, v, w))
        
        # M7 = (A12-A22)(B21+B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+0)*m + (j0+1)] = 1; u[(i0+1)*m + (j0+1)] = -1
        v[(j0+1)*p + (k0+0)] = 1; v[(j0+1)*p + (k0+1)] = 1
        w[(k0+0)*n + (i0+0)] = 1; w[(k0+1)*n + (i0+0)] = 1
        terms.append((u, v, w))
        
        return terms

    # Schoolbook term for 1x1x1
    def schoolbook_1x1(i0, j0, k0, n, m, p):
        u, v, w = make_term(i0, j0, k0, n, m, p)
        return [(u, v, w)]

    # Assemble 3x3x3
    # Split into 2x2 block at (0,0,0) and 1x1 blocks
    terms_333 = []
    # 2x2 block for C[0:2, 0:2]
    # This covers i in 0,1; j in 0,1; k in 0,1
    terms_333.extend(strassen_2x2(0, 0, 0, n, m, p))
    
    # Remaining terms for 3x3x3:
    # C[0:2, 2] = A[0:2, 0:3] B[0:3, 2]
    # C[2, 0:2] = A[2, 0:3] B[0:3, 0:2]
    # C[2, 2] = A[2, 0:3] B[0:3, 2]
    
    # We need to account for the contributions to these parts.
    # The Strassen block handled i,j,k in 0,1.
    # We need to handle terms where i=2 or j=2 or k=2.
    
    # Let's just use schoolbook for the remaining parts to be safe and simple, 
    # but we already counted the 2x2x2 part.
    # Actually, the Strassen block computes C[0:2, 0:2] using A[0:2, 0:2] and B[0:2, 0:2].
    # It does NOT use A[0:2, 2], A[2, 0:2], etc.
    # So we need to add terms for the "cross" terms.
    
    # Terms for C[0:2, 2]:
    # i in 0,1; k=2; j in 0,1,2
    # A[i,j] B[j,2]
    for i in range(2):
        for j in range(3):
            u, v, w = make_term(i, j, 2, n, m, p)
            terms_333.append((u, v, w))
            
    # Terms for C[2, 0:2]:
    # i=2; k in 0,1; j in 0,1,2
    # A[2,j] B[j,k]
    for k in range(2):
        for j in range(3):
            u, v, w = make_term(2, j, k, n, m, p)
            terms_333.append((u, v, w))
            
    # Terms for C[2, 2]:
    # i=2; k=2; j in 0,1,2
    # A[2,j] B[j,2]
    for j in range(3):
        u, v, w = make_term(2, j, 2, n, m, p)
        terms_333.append((u, v, w))
        
    # Total rank for 3x3x3: 7 (Strassen) + 2*3 (C[0:2,2]) + 2*3 (C[2,0:2]) + 3 (C[2,2]) = 7 + 6 + 6 + 3 = 22.
    # Wait, C[2,2] is included in the loops above?
    # Loop 1: i in 0,1; k=2. (6 terms)
    # Loop 2: i=2; k in 0,1. (6 terms)
    # Loop 3: i=2; k=2. (3 terms)
    # Total 22.
    # This is better than 24!
    
    # Now for 3x3x2 (k=3,4)
    # We can use the same logic.
    # Split into 2x2 block for i,j in 0,1 and k in 3,4?
    # No, k is the output column.
    # For 3x3x2, the "k" dimension is size 2.
    # We can use Strassen on the 2x2 output block?
    # Strassen is for 2x2x2.
    # Here we have 3x3x2.
    # We can split A and B.
    # A is 3x3. B is 3x2.
    # We can use Strassen for the 2x2 part of the output?
    # C[0:2, 0:2] = A[0:2, 0:3] B[0:3, 0:2].
    # This is 2x3x2.
    # We can split A[0:2, 0:3] into A[0:2, 0:2] and A[0:2, 2].
    # C[0:2, 0:2] = A[0:2, 0:2] B[0:2, 0:2] + A[0:2, 2] B[2, 0:2].
    # A[0:2, 0:2] B[0:2, 0:2] is 2x2x2. Rank 7.
    # A[0:2, 2] B[2, 0:2] is 2x1x2. Rank 4.
    # Total 11.
    
    # C[2, 0:2] = A[2, 0:3] B[0:3, 0:2].
    # = A[2, 0:2] B[0:2, 0:2] + A[2, 2] B[2, 0:2].
    # A[2, 0:2] B[0:2, 0:2] is 1x2x2. Rank 4.
    # A[2, 2] B[2, 0:2] is 1x1x2. Rank 2.
    # Total 6.
    
    # Total rank for 3x3x2: 11 + 6 = 17.
    
    terms_332 = []
    # 2x2x2 block for C[0:2, 0:2] (k=3,4)
    # i in 0,1; j in 0,1; k in 3,4
    terms_332.extend(strassen_2x2(0, 0, 3, n, m, p))
    
    # Cross terms for C[0:2, 0:2] involving j=2
    # i in 0,1; k in 3,4; j=2
    for i in range(2):
        for k in range(2):
            u, v, w = make_term(i, 2, k+3, n, m, p)
            terms_332.append((u, v, w))
            
    # Terms for C[2, 0:2]
    # i=2; k in 3,4; j in 0,1,2
    for k in range(2):
        for j in range(3):
            u, v, w = make_term(2, j, k+3, n, m, p)
            terms_332.append((u, v, w))
            
    # Total rank for 3x3x2: 7 + 4 + 6 = 17.
    
    # Combine 3x3x3 and 3x3x2
    all_terms = terms_333 + terms_332
    
    rank = len(all_terms)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u, v, w) in enumerate(all_terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
