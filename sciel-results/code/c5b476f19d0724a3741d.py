import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    
    The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For matrix multiplication, T[i, j, k] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Reshape U, V, W to tensor form for efficient contraction
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # We can compute this as a batched matrix multiplication or einsum
    # einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _rank_of(U, V, W):
    return U.shape[1] if U.size > 0 else 0

def _make_identity_decomposition(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _polish(U, V, W, n, m, p, max_iters=50, tol=1e-6):
    """
    Try to reduce the error using a simple gradient descent or alternating least squares.
    Since the objective is non-convex, this is a heuristic.
    """
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # We want to minimize || T - 1 ||^2 where T = U V W
    # Alternating least squares:
    # Fix V, W, solve for U.
    # This is a linear system for U if we view T as linear in U.
    # T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
    # Let M[j,k,i,c] = V[j,k,c] * W[k,i,c]
    # Then T[i,j,k] = sum_c U[i,j,c] M[j,k,i,c]
    # This is a linear system for each (i,j) if we flatten the other indices.
    
    # However, simple ALS might be slow or unstable.
    # Given the small size, we can try a few steps of random perturbation + local search.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.sum((np.einsum('ijc,jkc,kic->ijk', best_U.reshape(n,m,rank), best_V.reshape(m,p,rank), best_W.reshape(p,n,rank)) - 1.0)**2)
    
    for _ in range(max_iters):
        # Try a small random perturbation
        step = 0.1
        U_new = best_U + step * np.random.randn(*best_U.shape)
        V_new = best_V + step * np.random.randn(*best_V.shape)
        W_new = best_W + step * np.random.randn(*best_W.shape)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_new.reshape(n,m,rank), V_new.reshape(m,p,rank), W_new.reshape(p,n,rank))
        err = np.sum((T - 1.0)**2)
        
        if err < best_err:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_err = err
            if best_err < tol:
                break
        else:
            step *= 0.5
    
    # Snap to nearest half-integer
    U_snap = np.round(best_U * 2) / 2
    V_snap = np.round(best_V * 2) / 2
    W_snap = np.round(best_W * 2) / 2
    
    if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
        return U_snap, V_snap, W_snap, True
    
    return best_U, best_V, best_W, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Start with the known schoolbook decomposition
    best_result = _make_identity_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Verify the initial decomposition
    assert _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p), \
        "Initial decomposition failed verification"
    
    rng = random.Random(12345)
    np.random.seed(42)
    
    # Try to find a better (lower rank) decomposition
    # We try ranks from 39 down to 30
    for target_rank in range(n*m*p - 1, 29, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random initializations
        attempts = 10
        for _ in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with small random integers/half-integers
            # This helps in finding structured solutions
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
            
            # Polish
            U_pol, V_pol, W_pol, success = _polish(U, V, W, n, m, p)
            
            if success:
                best_result = {"U": U_pol, "V": V_pol, "W": W_pol}
                best_rank = target_rank
                break
        
        if best_rank < n*m*p:
            break
    
    return best_result
