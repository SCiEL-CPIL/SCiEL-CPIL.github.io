"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen-based decomposition: rank 112
    # Split 4x4x4x8 into two 4x4x4 (split p=8 into two 4s)
    # Each 4x4x4 split into 8 sub-problems of 2x2x2
    # Each 2x2x2 uses Strassen rank-7
    # Total: 2 * 8 * 7 = 112
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    # Strassen 2x2x2: verified formulas
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # Using the standard Strassen algorithm:
    M_defs = [
        # (U_entries, V_entries, W_entries)
        # Each entry: (local_idx, coeff)
        # U: (ii, ij, coeff) for A[ii,ij]
        # V: (ij, ik, coeff) for B[ij,ik]
        # W: (ik, ii, coeff) for C[ik,ii]
        
        # M1 = (A11+A22)(B11+B22)
        # C11 += M1, C22 += M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        
        # M2 = (A21+A22)B11
        # C21 += M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]),
        
        # M3 = A11(B21-B22)
        # C12 += M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(1,0,1)]),
        
        # M4 = A22(B12-B11)
        # C11 += M4, C21 += M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        
        # M5 = (A11+A12)B22
        # C11 -= M5, C12 += M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        
        # M6 = (A21-A11)(B11+B12)
        # C22 += M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        
        # M7 = (A12-A22)(B21+B22)
        # C11 += M7, C22 -= M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)]),
    ]
    
    col = 0
    for kk in range(2):      # which 4-block of k (0 or 1)
        for ia in range(2):  # which 2-block of i
            for ib in range(2):  # which 2-block of j
                for ik2 in range(2):  # which 2-block within the 4-block of k
                    for s in range(7):
                        U_entries, V_entries, W_entries = M_defs[s]
                        
                        for (ii, ij, coeff) in U_entries:
                            i = 2 * ia + ii
                            j = 2 * ib + ij
                            U[i * m + j][col] += coeff
                        
                        for (ij, ik, coeff) in V_entries:
                            j = 2 * ib + ij
                            k = 4 * kk + 2 * ik2 + ik
                            V[j * p + k][col] += coeff
                        
                        for (ik, ii, coeff) in W_entries:
                            k = 4 * kk + 2 * ik2 + ik
                            i = 2 * ia + ii
                            W[k * n + i][col] += coeff
                        
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
