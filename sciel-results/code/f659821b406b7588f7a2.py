import time
import numpy as np
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Construct Strassen-based decomposition
    # Split j (m=4) into 2 blocks of size 2 (alpha = 0, 1)
    # Split k (p=8) into 4 blocks of size 2 (gamma = 0, 1, 2, 3)
    # For each (alpha, gamma), we have a 2x2 x 2x2 multiplication
    # Strassen uses 7 multiplications.
    # Total rank = 2 * 4 * 7 = 56.
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for alpha in range(2):
        # j indices: 2*alpha, 2*alpha + 1
        j0 = 2 * alpha
        j1 = 2 * alpha + 1
        
        for gamma in range(4):
            # k indices: 4*gamma, 4*gamma + 1
            k0 = 4 * gamma
            k1 = 4 * gamma + 1
            
            # Strassen products P1..P7
            # P1 = (X00 + X11)(Y00 + Y11)
            # X00 = A[0, j0], X11 = A[1, j1]
            # Y00 = B[j0, k0], Y11 = B[j1, k1]
            # Contributes to Z00 (C[0, k0]) +1, Z11 (C[1, k1]) +1
            # Wait, Z11 in Strassen is C[1, k1]?
            # Local Z[1,1] corresponds to global i=1, k=k1.
            # Yes.
            
            # P1
            U[0*m + j0, r] = 1
            U[1*m + j1, r] = 1
            V[j0*p + k0, r] = 1
            V[j1*p + k1, r] = 1
            W[k0*n + 0, r] = 1 # Z00
            W[k1*n + 1, r] = 1 # Z11
            r += 1
            
            # P2 = (X21 + X22)Y11
            # X21 = A[1, j0], X22 = A[1, j1]
            # Y11 = B[j1, k1]
            # Contributes to Z21 (C[1, k0]) +1?
            # Strassen: Z21 = P2 + P4.
            # Local Z[1,0] is C[1, k0].
            # So W[k0*n + 1, r] = 1.
            U[1*m + j0, r] = 1
            U[1*m + j1, r] = 1
            V[j1*p + k1, r] = 1
            W[k0*n + 1, r] = 1
            r += 1
            
            # P3 = X11(Y12 - Y22)
            # X11 = A[1, j1]
            # Y12 = B[j0, k1], Y22 = B[j1, k1]
            # Contributes to Z12 (C[0, k1]) +1
            # Strassen: Z12 = P3 + P5.
            # Local Z[0,1] is C[0, k1].
            # So W[k1*n + 0, r] = 1.
            U[1*m + j1, r] = 1
            V[j0*p + k1, r] = 1
            V[j1*p + k1, r] = -1
            W[k1*n + 0, r] = 1
            r += 1
            
            # P4 = X22(Y21 - Y11)
            # X22 = A[1, j1]
            # Y21 = B[j1, k0], Y11 = B[j0, k0]
            # Contributes to Z21 (C[1, k0]) +1 and Z00 (C[0, k0]) +1?
            # Strassen: Z00 = P1 + P4 - P5 + P7.
            # Z21 = P2 + P4.
            # So W[k0*n + 0, r] = 1 and W[k0*n + 1, r] = 1.
            U[1*m + j1, r] = 1
            V[j1*p + k0, r] = 1
            V[j0*p + k0, r] = -1
            W[k0*n + 0, r] = 1
            W[k0*n + 1, r] = 1
            r += 1
            
            # P5 = (X11 + X12)Y22
            # X11 = A[1, j1], X12 = A[0, j1]
            # Y22 = B[j1, k1]
            # Contributes to Z12 (C[0, k1]) +1 and Z00 (C[0, k0]) -1?
            # Strassen: Z12 = P3 + P5.
            # Z00 = P1 + P4 - P5 + P7.
            # So W[k1*n + 0, r] = 1 and W[k0*n + 0, r] = -1.
            U[1*m + j1, r] = 1
            U[0*m + j1, r] = 1
            V[j1*p + k1, r] = 1
            W[k1*n + 0, r] = 1
            W[k0*n + 0, r] = -1
            r += 1
            
            # P6 = (X21 - X11)(Y11 + Y12)
            # X21 = A[1, j0], X11 = A[1, j1]
            # Y11 = B[j0, k0], Y12 = B[j0, k1]
            # Contributes to Z22 (C[1, k1]) +1?
            # Strassen: Z22 = P1 - P2 + P3 + P6.
            # Local Z[1,1] is C[1, k1].
            # So W[k1*n + 1, r] = 1.
            U[1*m + j0, r] = 1
            U[1*m + j1, r] = -1
            V[j0*p + k0, r] = 1
            V[j0*p + k1, r] = 1
            W[k1*n + 1, r] = 1
            r += 1
            
            # P7 = (X12 - X22)(Y21 + Y22)
            # X12 = A[0, j1], X22 = A[1, j1]
            # Y21 = B[j1, k0], Y22 = B[j1, k1]
            # Contributes to Z00 (C[0, k0]) +1?
            # Strassen: Z00 = P1 + P4 - P5 + P7.
            # So W[k0*n + 0, r] = 1.
            U[0*m + j1, r] = 1
            U[1*m + j1, r] = -1
            V[j1*p + k0, r] = 1
            V[j1*p + k1, r] = 1
            W[k0*n + 0, r] = 1
            r += 1
            
    # Verify
    # Check if entries are integers (they are +/-1 or 0)
    # Check decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
