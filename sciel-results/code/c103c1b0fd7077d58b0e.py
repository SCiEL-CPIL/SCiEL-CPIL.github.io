import numpy as np
import time
import random

def _build_rank17():
    """Construct a rank-17 decomposition using Strassen on 2x2x2 sub-block."""
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n*m, rank))  # (6, 17)
    V = np.zeros((m*p, rank))  # (9, 17)
    W = np.zeros((p*n, rank))  # (6, 17)
    
    # Block 1: Strassen for (i∈{0,1}, j∈{0,1}, k∈{0,1}) - 7 terms
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[3, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    
    # M4: A11(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Block 2: 2x1x2 for (i∈{0,1}, j=2, k∈{0,1}) - 4 terms
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[1, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[2, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Block 3: 2x2x1 for (i∈{0,1}, j∈{0,1}, k=2) - 4 terms
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # Block 4: 2x1x1 for (i∈{0,1}, j=2, k=2) - 2 terms
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=3, p=3):
    """Verify that the decomposition is exact."""
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    return np.allclose(result, target, atol=1e-10)

def _try_rank16_search(max_time=90):
    """Try to find a rank-16 decomposition using gradient descent."""
    n, m, p = 2, 3, 3
    start = time.time()
    deadline = start + max_time
    
    best_error = 1e10
    best_solution = None
    
    for attempt in range(80):
        if time.time() > deadline - 3:
            break
        
        np.random.seed(attempt * 7919 + 31337)
        scale = 0.5 + 0.5 * (attempt % 4)
        U = np.random.randn(n*m, 16) * scale
        V = np.random.randn(m*p, 16) * scale
        W = np.random.randn(p*n, 16) * scale
        
        lr = 0.005 + 0.002 * (attempt % 3)
        for step in range(1500):
            if time.time() > deadline - 1:
                break
            
            U_r = U.reshape(n, m, 16)
            V_r = V.reshape(m, p, 16)
            W_r = W.reshape(p, n, 16)
            
            result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            target = np.ones((n, m, p))
            diff = result - target
            error = float(np.sum(diff**2))
            
            if error < 1e-10:
                for mult in [1, 2, 4]:
                    U_rnd = np.round(U * mult) / mult
                    V_rnd = np.round(V * mult) / mult
                    W_rnd = np.round(W * mult) / mult
                    if _verify_decomposition(U_rnd, V_rnd, W_rnd):
                        return U_rnd, V_rnd, W_rnd
                break
            
            grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r).reshape(n*m, 16)
            grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r).reshape(m*p, 16)
            grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r).reshape(p*n, 16)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if error > best_error:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.03)
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
    
    if best_solution is not None and best_error < 0.05:
        U, V, W = best_solution
        for mult in [1, 2, 4, 8]:
            U_rnd = np.round(U * mult) / mult
            V_rnd = np.round(V * mult) / mult
            W_rnd = np.round(W * mult) / mult
            if _verify_decomposition(U_rnd, V_rnd, W_rnd):
                return U_rnd, V_rnd, W_rnd
    
    return None

def construct():
    start_time = time.time()
    
    # Try rank 16 first
    sol16 = _try_rank16_search(max_time=100)
    if sol16 is not None:
        U, V, W = sol16
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _build_rank17()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
