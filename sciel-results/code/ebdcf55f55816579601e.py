import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 14 decomposition using two Strassen 2x2 blocks
    # U: 4 x 14
    # V: 8 x 14
    # W: 8 x 14
    
    U = np.zeros((nm, 14))
    V = np.zeros((mp, 14))
    W = np.zeros((pn, 14))
    
    # Strassen 2x2 products for A (2x2) and B_block (2x2)
    # A indices: 0:a00, 1:a01, 2:a10, 3:a11
    # B indices for block 1: 0:b00, 1:b01, 2:b10, 3:b11
    # B indices for block 2: 4:b02, 5:b03, 6:b12, 7:b13
    
    # Product 1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1 # Wait, V index for b11 is 3?
    # V indices:
    # 0: b00 (row 0 col 0)
    # 1: b01 (row 0 col 1)
    # 2: b02 (row 0 col 2)
    # 3: b03 (row 0 col 3)
    # 4: b10 (row 1 col 0)
    # 5: b11 (row 1 col 1)
    # 6: b12 (row 1 col 2)
    # 7: b13 (row 1 col 3)
    
    # Correct V indices for Strassen block 1 (cols 0,1):
    # b00 -> 0
    # b01 -> 1
    # b10 -> 4
    # b11 -> 5
    
    # Correct V indices for Strassen block 2 (cols 2,3):
    # b02 -> 2
    # b03 -> 3
    # b12 -> 6
    # b13 -> 7
    
    # W indices:
    # W[k*n+i]
    # Block 1 (cols 0,1):
    # C[0,0] -> k=0, i=0 -> 0
    # C[1,0] -> k=0, i=1 -> 1
    # C[0,1] -> k=1, i=0 -> 2
    # C[1,1] -> k=1, i=1 -> 3
    # Block 2 (cols 2,3):
    # C[0,2] -> k=2, i=0 -> 4
    # C[1,2] -> k=2, i=1 -> 5
    # C[0,3] -> k=3, i=0 -> 6
    # C[1,3] -> k=3, i=1 -> 7
    
    # Re-define the 14 products with correct indices
    
    # Block 1
    # r=0: M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C[0,0] and C[1,1] get M1
    
    # r=1: M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C[1,0] gets M2, C[1,1] gets -M2
    
    # r=2: M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1 # C[0,1] gets M3, C[1,1] gets M3
    
    # r=3: M4 = a11(b10-b00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1 # C[0,0] gets M4, C[1,0] gets M4
    
    # r=4: M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C[0,0] gets -M5, C[0,1] gets M5
    
    # r=5: M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C[1,1] gets M6
    
    # r=6: M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1 # C[0,0] gets M7
    
    # Block 2
    # r=7: N1 = (a00+a11)(b02+b13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1 # C[0,2] and C[1,3]
    
    # r=8: N2 = (a10+a11)b02
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1 # C[1,2] and C[1,3]
    
    # r=9: N3 = a00(b03-b13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1 # C[0,3] and C[1,3]
    
    # r=10: N4 = a11(b12-b02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1 # C[0,2] and C[1,2]
    
    # r=11: N5 = (a00+a01)b13
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1 # C[0,2] and C[0,3]
    
    # r=12: N6 = (a10-a00)(b02+b03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1 # C[1,3]
    
    # r=13: N7 = (a01-a11)(b12+b13)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1 # C[0,2]
    
    # Verify
    # T_ijk = sum_r U_i_r V_j_r W_k_r
    # We need T_ijk = 1 if j is the middle index matching A and B, else 0?
    # No, the tensor is defined such that sum_r U[i*m+j][r] V[j*p+k][r] W[k*n+i][r] = 1
    # for all i,j,k.
    
    # Let's verify using numpy
    # Construct the tensor T of shape (nm, mp, pn)
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    
    # This is a batched matrix multiplication or einsum
    # T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Check if T matches the identity tensor
    # Identity tensor E[i, j, k] is 1 if i, j, k form a valid multiplication path?
    # No, the condition is:
    # For each i in [0, n), j in [0, m), k in [0, p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Let's check this specific condition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to naive
        rank = n * m * p
        U_naive = np.zeros((nm, rank))
        V_naive = np.zeros((mp, rank))
        W_naive = np.zeros((pn, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j, r] = 1
                    V_naive[j*p+k, r] = 1
                    W_naive[k*n+i, r] = 1
                    r += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}

