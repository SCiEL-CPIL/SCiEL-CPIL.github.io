"""Matrix multiplication 4,4,9 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]
n, m, p = DIMENSIONS
NM = n * m  # 16
MP = m * p  # 36
PN = p * n  # 36
SCHOOLBOOK_RANK = n * m * p  # 144

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0  # 10s safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _time_remaining():
    return TIME_LIMIT - _elapsed()

def _build_schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, tol=1e-9):
    """
    Verify that U, V, W (as numpy arrays) satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} (for all i,j,k)
    where the tensor T[i,j,k] = delta_{i,k} (matrix multiplication tensor).
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # For each (i,j,k), compute sum over c
    # T[i,j,k] should be 1 if i==k else 0
    # We can compute this as: for each c, the outer product U[:,c] * V[:,c] * W[:,c]
    # But more efficiently:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's check a subset or all
    # Full check is O(rank * n * m * p) which is fine for small ranks
    errors = 0.0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                # sum over c
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
                if err > tol:
                    errors += 1
    return max_err, errors

def _verify_fast(U, V, W, tol=1e-9):
    """Faster verification using numpy broadcasting."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Reshape for efficient computation
    # U has shape (NM, rank), V (MP, rank), W (PN, rank)
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Let's create arrays indexed properly
    # U_idx[i,j,c] = U[i*m+j, c]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a tensor contraction. Let's do it via einsum
    # U_r: (n, m, rank), V_r: (m, p, rank), W_r: (p, n, rank)
    # We want: sum over c and j? No, j is free.
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This can be computed as:
    # For each (i,j,k): sum_c U_r[i,j,c] * (V_r[j,k,c] * W_r[k,i,c])
    # Let's use einsum:
    # T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    # Expected: T[i,j,k] = 1 if i==k else 0
    expected = np.zeros((n, m, p))
    for i in range(n):
        expected[i, :, i] = 1.0
    max_err = np.max(np.abs(T - expected))
    return max_err

def _try_rank(rank, max_iters=200, seed=42):
    """
    Try to find a decomposition of the given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(NM, rank) * 0.1
    V = rng.randn(MP, rank) * 0.1
    W = rng.randn(PN, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Compute gradient-like updates
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want T[i,j,k] = delta_{i,k}
        
        # For each c, the contribution is the outer product of the three vectors
        # Let's compute the error tensor
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        expected = np.zeros((n, m, p))
        for i in range(n):
            expected[i, :, i] = 1.0
        
        error = T - expected
        err_norm = np.linalg.norm(error)
        
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err_norm < 1e-10:
            break
        
        # Gradient descent step
        # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] for each k
        # This is complex. Let's use a simpler approach:
        # Update each factor using the error
        # dU[i*m+j,c] = -lr * sum_k error[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
        # dV[j*p+k,c] = -lr * sum_i error[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
        # dW[k*n+i,c] = -lr * sum_j error[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
        
        lr = 0.01 / (1.0 + it * 0.01)
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = error[i, j, k]
                    if abs(e) > 1e-12:
                        for c in range(rank):
                            dU[i * m + j, c] += e * V[j * p + k, c] * W[k * n + i, c]
                            dV[j * p + k, c] += e * U[i * m + j, c] * W[k * n + i, c]
                            dW[k * n + i, c] += e * U[i * m + j, c] * V[j * p + k, c]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Final check
    max_err = _verify_fast(best_U, best_V, best_W)
    if max_err < 1e-8:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    # Start with schoolbook
    schoolbook = _build_schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bounds: For 4x4 * 4x9, the rank is at least max(16, 36, 36) = 36
    # But more sophisticated lower bounds apply. Let's try ranks from 36 upward.
    
    # First, try some known small ranks with random initialization
    # Use multiple seeds for robustness
    
    for rank in range(36, 80):
        if _time_remaining() < 10.0:
            break
        
        # Try multiple random initializations
        found = None
        for seed in range(10):
            if _time_remaining() < 5.0:
                break
            result = _try_rank(rank, max_iters=50, seed=seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            U, V, W = found
            # Round to nearest half-integer
            U = np.round(np.array(U) * 2) / 2
            V = np.round(np.array(V) * 2) / 2
            W = np.round(np.array(W) * 2) / 2
            # Verify after rounding
            max_err = _verify_fast(U, V, W)
            if max_err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    return schoolbook

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
