import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid tensor decomposition for n x m x p matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Compute product tensor P
    # P[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want P[i, j, k] == 1 for all i, j, k
    
    # Efficient check using einsum or loops
    # Given small sizes, loops are fine
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # sum over r
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if val != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    return U, V, W

def _find_decomposition(n, m, p, target_rank, time_limit=10):
    """
    Attempts to find a decomposition of rank target_rank for n x m x p tensor.
    Returns (U, V, W) or None.
    """
    start_time = time.time()
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    # Random restarts
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randint(-1, 2, (U_dim, target_rank))
        V = np.random.randint(-1, 2, (V_dim, target_rank))
        W = np.random.randint(-1, 2, (W_dim, target_rank))
        
        # Local search
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            # Check error
            err = 0
            for u_idx, v_idx, w_idx in indices:
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if val != 1:
                    err += 1
                    if err > 10: # Early exit if too many errors
                        break
            
            if err == 0:
                return U, V, W
                
            # Flip a random entry
            mat = random.choice([U, V, W])
            row = random.randint(0, mat.shape[0] - 1)
            col = random.randint(0, target_rank - 1)
            old_val = mat[row, col]
            mat[row, col] = random.randint(-1, 2)
            
    return None

def construct():
    n, m, p = 3, 3, 5
    
    # Split B into B1 (3x3) and B2 (3x2)
    # C1 = A B1 (3x3), C2 = A B2 (3x2)
    
    # Part 1: 3x3x3
    # Try to find rank 23
    U1, V1, W1 = _find_decomposition(3, 3, 3, 23, time_limit=30)
    if U1 is None:
        # Fallback to schoolbook
        U1, V1, W1 = _schoolbook_decomposition(3, 3, 3)
        
    # Part 2: 3x3x2
    # Try to find rank 17
    U2, V2, W2 = _find_decomposition(3, 3, 2, 17, time_limit=30)
    if U2 is None:
        # Fallback to schoolbook
        U2, V2, W2 = _schoolbook_decomposition(3, 3, 2)
        
    # Combine
    # U: (9, R1 + R2)
    U = np.hstack([U1, U2])
    
    # V: (15, R1 + R2)
    # V1 has non-zeros in rows corresponding to k=0,1,2 of B1.
    # In the full B (3x5), these are indices j*5 + k for k=0,1,2.
    # V1 is currently sized (9, R1) for B1 (3x3).
    # We need to map V1 rows to the full V rows.
    V_full = np.zeros((15, U.shape[1]))
    r1 = U1.shape[1]
    for j in range(3):
        for k in range(3):
            # V1 row index for B1[j, k] is j*3 + k
            # Full V row index for B[j, k] is j*5 + k
            V_full[j*5 + k, :r1] = V1[j*3 + k, :]
            
    # V2 is sized (6, R2) for B2 (3x2).
    # B2[j, k'] corresponds to B[j, 3+k'] in full B.
    r2 = U2.shape[1]
    for j in range(3):
        for k_prime in range(2):
            # V2 row index for B2[j, k'] is j*2 + k_prime
            # Full V row index for B[j, 3+k'] is j*5 + (3+k')
            V_full[j*5 + (3+k_prime), r1:r1+r2] = V2[j*2 + k_prime, :]
            
    V = V_full
    
    # W: (15, R1 + R2)
    # W1 has non-zeros in rows corresponding to k=0,1,2 of C1.
    # In full C (3x5), these are indices k*3 + i for k=0,1,2.
    # W1 is currently sized (9, R1) for C1 (3x3).
    W_full = np.zeros((15, U.shape[1]))
    for k in range(3):
        for i in range(3):
            # W1 row index for C1[i, k] is k*3 + i
            # Full W row index for C[i, k] is k*3 + i
            W_full[k*3 + i, :r1] = W1[k*3 + i, :]
            
    # W2 is sized (6, R2) for C2 (3x2).
    # C2[i, k'] corresponds to C[i, 3+k'] in full C.
    for k_prime in range(2):
        for i in range(3):
            # W2 row index for C2[i, k'] is k_prime*3 + i
            # Full W row index for C[i, 3+k'] is (3+k_prime)*3 + i
            W_full[(3+k_prime)*3 + i, r1:r1+r2] = W2[k_prime*3 + i, :]
            
    W = W_full
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # If verification fails (should not happen if sub-verifications passed and mapping is correct),
        # return schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
