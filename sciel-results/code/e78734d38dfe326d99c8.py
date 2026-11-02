"""Tensor decomposition for 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    def verify_full(U, V, W):
        """Full verification: check all (i',j',j'',k') combinations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = len(U_arr[0])
        # T[i',j',j'',k'] = sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
        # Must equal delta_{i',i} delta_{k',k} delta_{j',j''}
        # We check for all i, i_prime, j_prime, j_double_prime, k, k_prime
        for i in range(n):
            for i_prime in range(n):
                for j_prime in range(m):
                    for j_double_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                expected = 1.0 if (i_prime == i and k_prime == k and j_prime == j_double_prime) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += W_arr[k * n + i, r] * U_arr[i_prime * m + j_prime, r] * V_arr[j_double_prime * p + k_prime, r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True

    # Construct Rank 26 Decomposition
    rank = 26
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    
    # --- Strassen Core (Rank 7) for i,j,k in {0,1} ---
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # C00 += 1, C11 += 1
    U[0*3+0, 0] = 1.0; U[1*3+1, 0] = 1.0
    V[0*3+0, 0] = 1.0; V[1*3+1, 0] = 1.0
    W[0*3+0, 0] = 1.0; W[1*3+1, 0] = 1.0
    
    # Term 1: M2 = (A10+A11)B00
    # C10 += 1, C11 -= 1
    U[1*3+0, 1] = 1.0; U[1*3+1, 1] = 1.0
    V[0*3+0, 1] = 1.0
    W[0*3+1, 1] = 1.0; W[1*3+1, 1] = -1.0
    
    # Term 2: M3 = A00(B01-B11)
    # C01 += 1, C11 += 1
    U[0*3+0, 2] = 1.0
    V[0*3+1, 2] = 1.0; V[1*3+1, 2] = -1.0
    W[1*3+0, 2] = 1.0; W[1*3+1, 2] = 1.0
    
    # Term 3: M4 = A11(B10-B00)
    # C00 += 1, C10 += 1
    U[1*3+1, 3] = 1.0
    V[1*3+0, 3] = 1.0; V[0*3+0, 3] = -1.0
    W[0*3+0, 3] = 1.0; W[0*3+1, 3] = 1.0
    
    # Term 4: M5 = (A00+A01)B11
    # C00 -= 1, C01 += 1
    U[0*3+0, 4] = 1.0; U[0*3+1, 4] = 1.0
    V[1*3+1, 4] = 1.0
    W[0*3+0, 4] = -1.0; W[1*3+0, 4] = 1.0
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # C11 += 1
    U[1*3+0, 5] = 1.0; U[0*3+0, 5] = -1.0
    V[0*3+0, 5] = 1.0; V[0*3+1, 5] = 1.0
    W[1*3+1, 5] = 1.0
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # C00 += 1
    U[0*3+1, 6] = 1.0; U[1*3+1, 6] = -1.0
    V[1*3+0, 6] = 1.0; V[1*3+1, 6] = 1.0
    W[0*3+0, 6] = 1.0
    
    # --- Border Terms (Rank 19) ---
    # Standard terms for (i,j,k) where at least one index is 2
    
    border_terms = [
        (0,0,2), (0,1,2), (0,2,2),
        (1,0,2), (1,1,2), (1,2,2),
        (2,0,0), (2,1,0), (2,2,0),
        (2,0,1), (2,1,1), (2,2,1),
        (2,0,2), (2,1,2), (2,2,2)
    ]
    
    for idx, (i, j, k) in enumerate(border_terms):
        t = 7 + idx
        U[i*3+j, t] = 1.0
        V[j*3+k, t] = 1.0
        W[k*3+i, t] = 1.0
        
    # Verify
    if verify_full(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Schoolbook (Rank 27) if verification fails (should not happen)
    U27 = np.zeros((9, 27), dtype=float)
    V27 = np.zeros((9, 27), dtype=float)
    W27 = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U27[i*3+j, col] = 1.0
                V27[j*3+k, col] = 1.0
                W27[k*3+i, col] = 1.0
                col += 1
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
# EVOLVE-BLOCK-END
