"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import lstsq

def _schoolbook():
    """Return the schoolbook decomposition as numpy arrays."""
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-8):
    """Check if the decomposition exactly reconstructs the multiplication tensor."""
    n, m, p = DIMENSIONS
    # Build the full multiplication tensor T[i,j,k] = delta (it's 1 for each term)
    # The decomposition should satisfy:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For each (i,j,k) that doesn't correspond to a valid multiplication term:
    #   Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    #   where i in [0,n), j in [0,m), k in [0,p).
    # The decomposition must satisfy:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait, that's not right. Let me think more carefully.
    
    # The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Check all (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(U0, V0, W0, target_rank, max_iter=50, tol=1e-6, time_limit=None):
    """
    Try to reduce the rank by projecting to a lower-dimensional space.
    Uses iterative least squares to find a rank-target_rank decomposition.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize with random projection from the schoolbook
    rng = np.random.RandomState(42)
    U = np.zeros((dim_u, target_rank))
    V = np.zeros((dim_v, target_rank))
    W = np.zeros((dim_w, target_rank))
    
    # Initialize using random linear combinations of the schoolbook columns
    for r in range(target_rank):
        # Pick random weights for each factor
        w_u = rng.randn(len(U0[0])) * 0.1
        w_v = rng.randn(len(V0[0])) * 0.1
        w_w = rng.randn(len(W0[0])) * 0.1
        U[:, r] = U0 @ w_u
        V[:, r] = V0 @ w_v
        W[:, r] = W0 @ w_w
    
    # Iteratively refine using alternating least squares
    for it in range(max_iter):
        if time_limit is not None and (time.time() - time_limit[0]) > time_limit[1]:
            break
        
        # Update U: for each row i*m+j, solve for U[i*m+j, :] given V and W
        for i in range(n):
            for j in range(m):
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then: A @ U[i*m+j, :] = ones(p)
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                rhs = np.ones(p)
                try:
                    sol, _, _, _ = lstsq(A, rhs, rcond=tol)
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Update V: for each row j*p+k, solve for V[j*p+k, :] given U and W
        for j in range(m):
            for k in range(p):
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then: B @ V[j*p+k, :] = ones(n)
                B = np.zeros((n, target_rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                rhs = np.ones(n)
                try:
                    sol, _, _, _ = lstsq(B, rhs, rcond=tol)
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Update W: for each row k*n+i, solve for W[k*n+i, :] given U and V
        for k in range(p):
            for i in range(n):
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then: C @ W[k*n+i, :] = ones(m)
                C = np.zeros((m, target_rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                rhs = np.ones(m)
                try:
                    sol, _, _, _ = lstsq(C, rhs, rcond=tol)
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Check if we've converged
        if _verify(U, V, W, tol=1e-5):
            return U, V, W
    
    return None

def _to_list(arr):
    """Convert numpy array to list of lists with rounded values."""
    return [[float(round(val, 10)) for val in row] for row in arr]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition
    for multiplying a 2 by 4 matrix by a 4 by 7 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              U shape: (n*m, rank)
              V shape: (m*p, rank)
              W shape: (p*n, rank)
              where rank is minimized.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: 120s limit, use 120s
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to reduce rank from best_rank-1 down to some minimum
    # Theoretical lower bounds: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    min_possible = max(n*m, m*p, p*n)
    
    for target_rank in range(best_rank - 1, min_possible - 1, -1):
        if (time.time() - start_time) > deadline - 30:  # Leave 30s margin
            break
        
        result = _try_rank_reduction(U0, V0, W0, target_rank, max_iter=30, time_limit=[start_time, deadline - 30])
        if result is not None:
            U, V, W = result
            # Verify with tighter tolerance
            if _verify(U, V, W, tol=1e-8):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                # If we found a valid lower rank, continue trying to go lower
                # But we need to restart from this new solution
                U0, V0, W0 = U.copy(), V.copy(), W.copy()
    
    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}
# EVOLVE-BLOCK-END
