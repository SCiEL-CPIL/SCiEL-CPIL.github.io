import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Target rank 54
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2: (A10+A11)B00 -> C10+=1
    # M3: A00(B10-B11) -> C01+=1
    # M4: A11(B01-B00) -> C00+=1, C10+=1
    # M5: (A00+A10)B11 -> C00-=1, C01+=1
    # M6: (A11-A00)(B00+B01) -> C11+=1
    # M7: (A01-A11)(B10+B11) -> C00+=1, C11-=1
    
    # Format: list of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): coeff}
    # V_coeffs: dict {(j,k): coeff}
    # W_coeffs: dict {(k,i): coeff}
    
    strassen_terms = [
        # M1
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1}  # C10 -> k=0, i=1
        ),
        # M3
        (
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(1,0): 1}  # C01 -> k=1, i=0
        ),
        # M4
        (
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1} # C00 (k=0,i=0), C10 (k=0,i=1)
        ),
        # M5
        (
            {(0,0): 1, (1,0): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1} # C00 (k=0,i=0), C01 (k=1,i=0)
        ),
        # M6
        (
            {(1,1): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1} # C11 -> k=1, i=1
        ),
        # M7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): -1} # C00 (k=0,i=0), C11 (k=1,i=1)
        )
    ]
    
    def add_strassen_block(i_off, j_off, k_off, start_col):
        nonlocal col
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            c = start_col
            for (i, j), val in u_coeffs.items():
                U[i * m + (j_off + j), c] += val
            for (j, k), val in v_coeffs.items():
                V[(j_off + j) * p + (k_off + k), c] += val
            for (k, i), val in w_coeffs.items():
                W[(k_off + k) * n + i, c] += val
            start_col += 1
        return start_col

    # 6 Strassen blocks
    # j ranges: 0-1, 2-3
    # k ranges: 0-1, 2-3, 4-5
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            col = add_strassen_block(0, j_off, k_off, col)
            
    # 1 Trivial block for j=4, k=0..5
    # A is 2x1 (col 4), B is 1x6 (row 4)
    # C is 2x6
    # C[i,k] = A[i,4] * B[4,k]
    for i in range(2):
        for k in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Verify
    # T1 shape (10, 360)
    # T2 shape (30, 120)
    # T3 shape (12, 300)
    
    def khatri_rao(A, B):
        dim1, R = A.shape
        dim2, _ = B.shape
        prod = A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
        return prod.reshape(R, -1).T

    T1 = np.zeros((n*m, m*p * p*n))
    T2 = np.zeros((m*p, n*m * p*n))
    T3 = np.zeros((p*n, n*m * m*p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T1[a, b * (p*n) + c] = 1.0
                T2[b, a * (p*n) + c] = 1.0
                T3[c, a * (m*p) + b] = 1.0
                
    MWV = khatri_rao(W, V)
    T1_recon = U @ MWV.T
    if not np.allclose(T1_recon, T1, atol=1e-9):
        # Fallback to trivial if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
