import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split: n=3→(2,1), m=4→(2,2), p=6→(2,2,2)
    
    # Sub-problems:
    # 3x (2,4,2) with rank 14 each (Strassen on two (2,2,2) blocks)
    # 3x (1,4,2) with rank 8 each (naive)
    
    rank = 3*14 + 3*8  # 66
    
    U = [[0]*rank for _ in range(n*m)]  # (12, 66)
    V = [[0]*rank for _ in range(m*p)]  # (24, 66)
    W = [[0]*rank for _ in range(p*n)]  # (18, 66)
    
    col = 0
    
    # Strassen (2,2,2) helper: adds 7 columns
    # Local A is 2x2 at global rows i0:i0+2, cols j0:j0+2
    # Local B is 2x2 at global rows j0:j0+2, cols k0:k0+2
    # Local C is 2x2 at global rows i0:i0+2, cols k0:k0+2
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        # 7 Strassen terms
        # Each term: (linear form in A) * (linear form in B) -> contributes to C
        # Term format: (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (local_i, local_j, coeff)
        # V_coeffs: list of (local_j, local_k, coeff)
        # W_coeffs: list of (local_k, local_i, coeff)
        
        terms = [
            # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
            # C[0,0]+=1, C[1,1]+=1
            {
                'U': [(0,0,1), (1,1,1)],
                'V': [(0,0,1), (1,1,1)],
                'W': [(0,0,1), (1,1,1)]
            },
            # M2 = (A[1,0]+A[1,1])*B[0,0]
            # C[1,0]+=1
            {
                'U': [(1,0,1), (1,1,1)],
                'V': [(0,0,1)],
                'W': [(0,1,1)]
            },
            # M3 = A[0,0]*(B[0,1]-B[1,1])
            # C[0,1]+=1
            {
                'U': [(0,0,1)],
                'V': [(0,1,1), (1,1,-1)],
                'W': [(1,0,1)]
            },
            # M4 = A[1,1]*(B[1,0]-B[0,0])
            # C[1,0]+=1, C[1,1]+=1
            {
                'U': [(1,1,1)],
                'V': [(1,0,1), (0,0,-1)],
                'W': [(0,1,1), (1,1,1)]
            },
            # M5 = (A[0,0]+A[0,1])*B[1,1]
            # C[0,1]+=1, C[1,1]+=-1
            {
                'U': [(0,0,1), (0,1,1)],
                'V': [(1,1,1)],
                'W': [(1,0,1), (1,1,-1)]
            },
            # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
            # C[1,1]+=1
            {
                'U': [(1,0,1), (0,0,-1)],
                'V': [(0,0,1), (0,1,1)],
                'W': [(1,1,1)]
            },
            # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
            # C[0,0]+=1, C[1,1]+=1
            {
                'U': [(0,1,1), (1,1,-1)],
                'V': [(1,0,1), (1,1,1)],
                'W': [(0,0,1), (1,1,1)]
            },
        ]
        
        for term in terms:
            for (li, lj, c) in term['U']:
                U[(i0+li)*m + (j0+lj)][col] += c
            for (lj, lk, c) in term['V']:
                V[(j0+lj)*p + (k0+lk)][col] += c
            for (lk, li, c) in term['W']:
                W[(k0+lk)*n + (i0+li)][col] += c
            col += 1
    
    # Naive (1,4,2) helper: adds 8 columns
    # Local A is 1x2 at global row i0, cols j0:j0+2
    # Local B is 2x2 at global rows j0:j0+2, cols k0:k0+2
    # Local C is 1x2 at global row i0, cols k0:k0+2
    def add_naive_142(i0, j0, k0):
        nonlocal col
        # A is 1x4 (but we split into two 1x2 blocks handled by two (1,2,2) naive)
        # Actually, for (1,4,2): A[0,j]*B[j,k] for j in 0..3, k in 0..1
        # Split j into 0-1 and 2-3
        # For j in 0-1: (1,2,2) naive, rank 4
        # For j in 2-3: (1,2,2) naive, rank 4
        # Total: 8
        
        # First half: j in 0-1
        for lj in range(2):
            for lk in range(2):
                U[i0*m + (j0+lj)][col] = 1
                V[(j0+lj)*p + (k0+lk)][col] = 1
                W[(k0+lk)*n + i0][col] = 1
                col += 1
        # Second half: j in 2-3
        for lj in range(2):
            for lk in range(2):
                U[i0*m + (j0+2+lj)][col] = 1
                V[(j0+2+lj)*p + (k0+lk)][col] = 1
                W[(k0+lk)*n + i0][col] = 1
                col += 1
    
    # Now add the 6 sub-problems
    # Top row (i in 0-1), 3 sub-problems for k in 0-1, 2-3, 4-5
    for ki in range(3):
        k0 = ki * 2
        # (2,4,2) = Strassen(A[0:2,0:2]*B[0:2,k0:k0+2]) + Strassen(A[0:2,2:4]*B[2:4,k0:k0+2])
        add_strassen_222(0, 0, k0)   # A[0:2,0:2] * B[0:2,k0:k0+2]
        add_strassen_222(0, 2, k0)   # A[0:2,2:4] * B[2:4,k0:k0+2]
    
    # Bottom row (i=2), 3 sub-problems for k in 0-1, 2-3, 4-5
    for ki in range(3):
        k0 = ki * 2
        add_naive_142(2, 0, k0)
    
    return {"U": U, "V": V, "W": W}
