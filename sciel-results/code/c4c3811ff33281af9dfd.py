"""Matrix multiplication 2,4,6 — rank-minimal exact tensor decomposition."""

import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W for Strassen's 2x2x2 matrix multiplication, rank 7."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # p1 = (a+d)(e+h)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # p2 = (c+d)e
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # p3 = a(f-h)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # p4 = d(g-e)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # p5 = (a+b)h
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # p6 = (c-a)(e+f)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # p7 = (b-d)(g+h)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0; W[1, 6] = 1.0
    
    return U, V, W

def _construct_rank42():
    """Construct a rank-42 decomposition using block structure."""
    n, m, p = 2, 4, 6
    dim_a = n * m  # 8
    dim_b = m * p  # 24
    dim_c = p * n  # 12
    
    rank = 42
    U = np.zeros((dim_a, rank), dtype=np.float64)
    V = np.zeros((dim_b, rank), dtype=np.float64)
    W = np.zeros((dim_c, rank), dtype=np.float64)
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    col = 0
    for a_block in range(2):  # A1 (cols 0-1) or A2 (cols 2-3)
        for b_sub in range(3):  # B sub-blocks (cols 0-1, 2-3, 4-5)
            for s in range(7):  # Strassen's 7 products
                r = col
                
                # A indices
                for idx in range(4):
                    U_val = U_str[idx, s]
                    if U_val != 0:
                        i = idx // 2
                        j_local = idx % 2
                        j_global = a_block * 2 + j_local
                        U[i * m + j_global, r] += U_val
                
                # B indices
                for idx in range(4):
                    V_val = V_str[idx, s]
                    if V_val != 0:
                        i_local = idx // 2
                        k_local = idx % 2
                        i_global = a_block * 2 + i_local
                        k_global = b_sub * 2 + k_local
                        V[i_global * p + k_global, r] += V_val
                
                # C indices
                for idx in range(4):
                    W_val = W_str[idx, s]
                    if W_val != 0:
                        i_out = idx // 2
                        k_out_local = idx % 2
                        k_out_global = b_sub * 2 + k_out_local
                        W[k_out_global * n + i_out, r] += W_val
                
                col += 1
    
    return U, V, W

def _validate(U, V, W, T, tol=1e-6):
    """Check if U, V, W exactly reconstruct T."""
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_hat, T, atol=tol)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _try_random_search(rank, time_limit=5.0):
    """Attempt to find a decomposition of the given rank using random sparse search."""
    n, m, p = 2, 4, 6
    T = _build_tensor(n, m, p)
    start_time = time.time()
    np.random.seed(42)
    
    # We try a few iterations of random sparse integer matrices
    # This is unlikely to succeed for rank < 42 if 42 is minimal, 
    # but it's a structural addition.
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # Initialize with random -1, 0, 1
        for c in range(rank):
            for idx in range(n * m):
                if random.random() < 0.2:
                    U[idx, c] = random.choice([-1, 1])
            for idx in range(m * p):
                if random.random() < 0.2:
                    V[idx, c] = random.choice([-1, 1])
            for idx in range(p * n):
                if random.random() < 0.2:
                    W[idx, c] = random.choice([-1, 1])
        
        if _validate(U, V, W, T, tol=1e-6):
            return U, V, W
            
    return None

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Try to find a lower rank decomposition via random search (short time limit)
    # We try rank 41 and 40.
    for target_rank in [41, 40]:
        if time.time() > deadline - 10:
            break
        result = _try_random_search(target_rank, time_limit=5.0)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 2. Fall back to the constructive rank-42 decomposition
    U, V, W = _construct_rank42()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
