"""Tensor rank minimization for 3x5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE_SECONDS = 140.0
SEED = 42

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid product term.
    For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
    Tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 for all valid (i,j,k).
    """
    T = np.ones((n, m, p))
    return T

def _decompose_check(U, V, W, n, m, p, tol=1e-7):
    """Check if U,V,W with shapes (n*m, r), (m*p, r), (p*n, r) decompose the tensor.
    W[k*n+i, r] convention: W is indexed by (k, i) where k is the column of B / row of C,
    i is the row of A / row of C.
    
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    """
    r = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    
    # Compute T_hat[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T_hat = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    T_true = np.ones((n, m, p))
    err = np.max(np.abs(T_hat - T_true))
    return err, T_hat

def _random_init(n, m, p, rank, rng):
    """Initialize U, V, W with random values."""
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    return U, V, W

def _gradient_step(U, V, W, n, m, p, lr, rng):
    """One step of gradient descent to minimize ||T_hat - T_true||_F^2.
    T_hat[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    """
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    
    T_true = np.ones((n, m, p))
    T_hat = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    E = T_hat - T_true  # error tensor
    
    # Gradient w.r.t. U[i*m+j, r] = sum_{k} E[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
    # grad_U[i,j,r] = sum_k E[i,j,k] * Vr[j,k,r] * Wr[k,i,r]
    grad_U = np.einsum('ijk,jkr,kir->ijr', E, Vr, Wr)
    grad_V = np.einsum('ijk,ijr,kir->jkr', E, Ur, Wr)
    grad_W = np.einsum('ijk,ijr,jkr->kir', E, Ur, Vr)
    
    U -= lr * grad_U.reshape(n * m, r)
    V -= lr * grad_V.reshape(m * p, r)
    W -= lr * grad_W.reshape(p * n, r)
    
    return U, V, W

def _try_rank(n, m, p, rank, rng, time_budget):
    """Try to find a rank-`rank` decomposition. Returns (U, V, W) or None."""
    start_time = time.time()
    max_iters = 5000
    
    U, V, W = _random_init(n, m, p, rank, rng)
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        U, V, W = _gradient_step(U, V, W, n, m, p, lr, rng)
        
        # Compute error
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        T_hat = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T_hat - np.ones((n, m, p))))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if err < 0.1:
            lr = 0.001
        elif err < 1.0:
            lr = 0.005
        else:
            lr = 0.01
        
        if best_err < 1e-7:
            break
    
    if best_err < 1e-7:
        return best_U, best_V, best_W
    return None

def _round_to_half(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    
    # Try ranks from low to high
    # Theoretical lower bound for 3x5x5 is at least max(n*m, m*p, p*n) = max(15,25,15) = 25
    # But that's not tight. Let's try from a reasonable lower bound.
    # Known: rank of matrix mult tensor is between max(n*m, m*p, p*n) and n*m*p
    # For 3x5x5: between 25 and 75.
    
    for rank in range(25, best_rank):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        time_budget = min(10.0, DEADLINE_SECONDS - 5 - (time.time() - start_time))
        if time_budget < 1.0:
            break
        
        result = _try_rank(n, m, p, rank, np_rng, time_budget)
        if result is not None:
            U, V, W = result
            # Try rounding to half-integers
            U_r, V_r, W_r = _round_to_half(U, V, W)
            err_r, _ = _decompose_check(U_r, V_r, W_r, n, m, p)
            if err_r < 1e-6:
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
            else:
                # Use unrounded if it's valid
                err_u, _ = _decompose_check(U, V, W, n, m, p)
                if err_u < 1e-7:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
