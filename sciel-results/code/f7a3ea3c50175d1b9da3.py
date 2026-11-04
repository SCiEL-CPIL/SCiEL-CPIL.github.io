import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

def verify(U, V, W, n, m, p, rank):
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == delta
    # Actually, the tensor is T[i,j,k] = 1
    # So sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1?
    # No, the tensor is T = sum e_ij e_jk e_ki
    # So for each i,j,k, the sum should be 1?
    # Wait, the decomposition is sum_r u_r v_r w_r = T
    # T[i,j,k] = 1 for all i,j,k?
    # No, T is the matrix multiplication tensor.
    # T[i,j,k] = 1 if the indices match the multiplication.
    # Actually, T[i,j,k] = 1 for all i,j,k is WRONG.
    # The tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # So T[i,j,k] = 1 for all i,j,k.
    # Yes, because the term A_{i,j} B_{j,k} contributes to C_{k,i}.
    # So for a fixed i,j,k, the coefficient is 1.
    # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try rank 20
    rank = 20
    for _ in range(1000):
        if time.time() > deadline:
            break
        U = np.random.randint(0, 2, (n*m, rank))
        V = np.random.randint(0, 2, (m*p, rank))
        W = np.random.randint(0, 2, (p*n, rank))
        if verify(U, V, W, n, m, p, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
