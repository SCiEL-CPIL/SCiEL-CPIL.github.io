"""Matrix multiplication tensor decomposition for 3x4x7 with rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is exact: sum_c U[:,c] outer V[:,c] outer W[:,c] == tensor."""
    rank = U.shape[1]
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    target = np.ones((n, m, p))
    return np.max(np.abs(tensor - target)) < tol


def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(2.0 * x) / 2.0


def _als_update_U(V, W, n, m, p, r):
    """
    Fix V and W, solve for U using minimum-norm least squares.
    For each (i,j): sum_c U[i*m+j, c] * A[i*m+j, c] = 1
    where A[i*m+j, c] = sum_k V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.zeros((n * m, r))
    for i in range(n):
        for j in range(m):
            # Compute coefficient vector A[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
            A = np.zeros(r)
            for k in range(p):
                A += V[j * p + k, :] * W[k * n + i, :]
            # Solve A @ U_row = 1 with minimum norm: U_row = A^T / (A @ A^T) if A != 0
            norm_sq = np.dot(A, A)
            if norm_sq > 1e-12:
                U[i * m + j, :] = A / norm_sq
            # else leave as zero (infeasible, will be caught by verification)
    return U


def _als_update_V(U, W, n, m, p, r):
    """
    Fix U and W, solve for V using minimum-norm least squares.
    For each (j,k): sum_c V[j*p+k, c] * B[j*p+k, c] = 1
    where B[j*p+k, c] = sum_i U[i*m+j, c] * W[k*n+i, c]
    """
    V = np.zeros((m * p, r))
    for j in range(m):
        for k in range(p):
            B = np.zeros(r)
            for i in range(n):
                B += U[i * m + j, :] * W[k * n + i, :]
            norm_sq = np.dot(B, B)
            if norm_sq > 1e-12:
                V[j * p + k, :] = B / norm_sq
    return V


def _als_update_W(U, V, n, m, p, r):
    """
    Fix U and V, solve for W using minimum-norm least squares.
    For each (k,i): sum_c W[k*n+i, c] * C[k*n+i, c] = 1
    where C[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    """
    W = np.zeros((p * n, r))
    for k in range(p):
        for i in range(n):
            C = np.zeros(r)
            for j in range(m):
                C += U[i * m + j, :] * V[j * p + k, :]
            norm_sq = np.dot(C, C)
            if norm_sq > 1e-12:
                W[k * n + i, :] = C / norm_sq
    return W


def _als_decompose(n, m, p, r, n_iter=100, seed=42):
    """
    Attempt to find a rank-r decomposition using ALS.
    Returns (U, V, W) or None if failed.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r)
    V = rng.randn(m * p, r)
    W = rng.randn(p * n, r)
    
    for it in range(n_iter):
        U_new = _als_update_U(V, W, n, m, p, r)
        V_new = _als_update_V(U_new, W, n, m, p, r)
        W_new = _als_update_W(U_new, V_new, n, m, p, r)
        
        # Check convergence
        if np.max(np.abs(U_new - U)) < 1e-10 and \
           np.max(np.abs(V_new - V)) < 1e-10 and \
           np.max(np.abs(W_new - W)) < 1e-10:
            U, V, W = U_new, V_new, W_new
            break
        
        U, V, W = U_new, V_new, W_new
    
    if _verify(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 84
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = target_rank
    
    # Try to find lower-rank decompositions using ALS
    # Try various ranks from low to high
    trial_ranks = [30, 36, 42, 48, 54, 60, 66, 72, 78]
    
    for trial_rank in trial_ranks:
        if time.time() > deadline - 15:
            break
        
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            result = _als_decompose(n, m, p, trial_rank, n_iter=50, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = trial_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < target_rank:
            break
    
    # Final polish: round to half-integers and re-verify
    if best_rank < target_rank:
        best_U_r = _round_to_half(best_U)
        best_V_r = _round_to_half(best_V)
        best_W_r = _round_to_half(best_W)
        if _verify(best_U_r, best_V_r, best_W_r, n, m, p):
            best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
    
    # Return as lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
