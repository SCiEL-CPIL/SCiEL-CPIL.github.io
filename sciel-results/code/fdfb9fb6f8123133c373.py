"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Identify disjoint 2x2x2 blocks for Strassen
    # I = {0, 1}
    # J = {0, 1}, {2, 3}
    # K = {0, 1}, {2, 3}, {4, 5}
    
    blocks = []
    I_set = [0, 1]
    J_sets = [[0, 1], [2, 3]]
    K_sets = [[0, 1], [2, 3], [4, 5]]
    
    for J in J_sets:
        for K in K_sets:
            blocks.append((I_set, J, K))
            
    # We will use Strassen for these 6 blocks
    # And schoolbook for the rest
    
    # Pre-allocate matrices
    # We don't know the exact rank yet, but it is rank_school - 6
    rank = rank_school - len(blocks)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
        col += 1
        
    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen(I, J, K):
        nonlocal col
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K
        
        # Strassen terms for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        # Term 1
        U[i0*m+j0, col] = 1; V[j0*p+k0, col] = 1; W[k0*n+i0, col] = 1
        U[i1*m+j1, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i1, col] = 1
        col += 1
        
        # Term 2
        U[i1*m+j0, col] = 1; V[j0*p+k0, col] = 1; W[k0*n+i1, col] = 1
        U[i1*m+j1, col] = 1; V[j1*p+k0, col] = 1; W[k0*n+i1, col] = 1
        col += 1
        
        # Term 3
        U[i0*m+j0, col] = 1; V[j1*p+k0, col] = 1; W[k0*n+i0, col] = 1
        U[i0*m+j0, col] = 1; V[j1*p+k1, col] = -1; W[k1*n+i0, col] = 1
        col += 1
        
        # Term 4
        U[i1*m+j1, col] = 1; V[j0*p+k1, col] = 1; W[k1*n+i1, col] = 1
        U[i1*m+j1, col] = 1; V[j0*p+k0, col] = -1; W[k0*n+i1, col] = 1
        col += 1
        
        # Term 5
        U[i0*m+j0, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i0, col] = 1
        U[i0*m+j1, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i0, col] = 1
        col += 1
        
        # Term 6
        U[i1*m+j0, col] = 1; V[j0*p+k0, col] = 1; W[k0*n+i1, col] = 1
        U[i0*m+j0, col] = -1; V[j0*p+k0, col] = 1; W[k0*n+i0, col] = 1
        U[i1*m+j0, col] = 1; V[j0*p+k1, col] = 1; W[k1*n+i1, col] = 1
        U[i0*m+j0, col] = -1; V[j0*p+k1, col] = 1; W[k1*n+i0, col] = 1
        col += 1
        
        # Term 7
        U[i0*m+j1, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i0, col] = 1
        U[i1*m+j1, col] = -1; V[j1*p+k1, col] = 1; W[k1*n+i1, col] = 1
        U[i0*m+j1, col] = 1; V[j1*p+k0, col] = 1; W[k0*n+i0, col] = 1
        U[i1*m+j1, col] = -1; V[j1*p+k0, col] = 1; W[k0*n+i1, col] = 1
        col += 1

    # Mark which (i,j,k) tuples are covered by Strassen blocks
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
    # Iterate through all schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    # Skip, handled by Strassen
                    continue
                else:
                    add_schoolbook(i, j, k)
                    
    # Add Strassen terms
    for I, J, K in blocks:
        add_strassen(I, J, K)
        
    # Convert to lists of lists for interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
