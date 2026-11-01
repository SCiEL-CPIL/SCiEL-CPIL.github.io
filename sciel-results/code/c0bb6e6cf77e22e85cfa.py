"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly represents 2x2 * 2x6 matrix multiplication.
    W[k*n+i, r] is the trace convention: row index = j*n + i where j is column of C, i is row of C.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Target tensor: T[i,k,j] = 1 (C[i,j] = sum_k A[i,k]*B[k,j])
    # Decomposition: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
    
    # Compute decomposed tensor efficiently using einsum
    # U_arr shape: (n*m, rank) -> reshape to (n, m, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T_decomp[i,k,j] = sum_r U3[i,k,r] * V3[k,j,r] * W3[j,i,r]
    T_decomp = np.einsum('ikr,kjr,jir->ikj', U3, V3, W3)
    
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We process 3 blocks of 2x2
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V row indices for this block
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]
        
        # W row indices for this block
        w_c00 = k0 * n + 0 # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n + 0 # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]
        
        # Strassen steps 0..6
        # Step 0: P1 = (a+d)(e+h)
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        # Step 1: P2 = (c+d)e
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        # Step 2: P3 = a(f-h)
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        # Step 3: P4 = d(g-e)
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        # Step 4: P5 = (a+b)h
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        # Step 5: P6 = (c-a)(e+f)
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        # Step 6: P7 = (b-d)(g+h)
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1

    # Verify correctness
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook rank 24 if Strassen verification fails
        rank24 = 24
        U24 = [[0] * rank24 for _ in range(n * m)]
        V24 = [[0] * rank24 for _ in range(m * p)]
        W24 = [[0] * rank24 for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U24[i*m+k][idx] = 1
                    V24[k*p+j][idx] = 1
                    W24[j*n+i][idx] = 1
                    idx += 1
        return {"U": U24, "V": V24, "W": W24}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
