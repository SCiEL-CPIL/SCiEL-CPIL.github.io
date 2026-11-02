"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _update_W(U, V, W, n, m, p, r):
    """Update W rows given U and V."""
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k*n + i, :] = sol

def _update_U(U, V, W, n, m, p, r):
    """Update U rows given V and W."""
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, r))
            for k in range(p):
                B[k, :] = V[j*p + k, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i*m + j, :] = sol

def _update_V(U, V, W, n, m, p, r):
    """Update V rows given U and W."""
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, r))
            for i in range(n):
                C[i, :] = U[i*m + j, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j*p + k, :] = sol

def _is_half_integer(mat, tol=1e-5):
    """Check if all entries are close to multiples of 0.5."""
    rounded = np.round(mat * 2) / 2
    return np.allclose(mat, rounded, atol=tol)

def _round_to_half_integer(mat):
    return np.round(mat * 2) / 2

def _compute_error(n, m, p, U, V, W):
    """Compute the total error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                err += (val - 1.0)**2
    return err

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s margin out of 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Initialize best with schoolbook solution
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1
                
    random.seed(42)
    np.random.seed(42)
    
    # Try structured initialization based on block decomposition
    # For 4x4 * 4x6, we can use 2x2 blocks
    # A is 4x4, B is 4x6, C is 4x6
    
    # Let's try to find a low-rank solution using a combination of
    # structured initialization and ALS refinement
    
    # First, try a few different structured approaches
    
    # Approach 1: Direct random search with better initialization
    for target_rank in range(80, 60, -4):
        if time.time() > deadline - 5:
            break
            
        attempts = 8
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with a mix of random and structured values
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            # Add some structure: make some entries zero to encourage sparsity
            for c in range(target_rank):
                if random.random() < 0.3:
                    U[random.randint(0, n*m-1), c] = 0
                if random.random() < 0.3:
                    V[random.randint(0, m*p-1), c] = 0
                if random.random() < 0.3:
                    W[random.randint(0, p*n-1), c] = 0
            
            # Normalize columns
            for c in range(target_rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-10:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U[:, c] *= scale
                    V[:, c] *= scale
                    W[:, c] *= scale
            
            # ALS iterations with early stopping
            prev_err = float('inf')
            for it in range(80):
                if time.time() > deadline - 5:
                    break
                    
                err = _compute_error(n, m, p, U, V, W)
                
                if err < 1e-8:
                    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                        if valid:
                            best_rank = target_rank
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                            found = True
                            break
                    elif _is_half_integer(U) or _is_half_integer(V) or _is_half_integer(W):
                        # Try rounding and see if it still works
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        valid, _ = _verify_rank(n, m, p, target_rank, U_rounded, V_rounded, W_rounded)
                        if valid:
                            best_rank = target_rank
                            best_U = U_rounded
                            best_V = V_rounded
                            best_W = W_rounded
                            found = True
                            break
                    else:
                        # Not half-integer yet, continue iterating
                        pass
                
                if err > prev_err * 1.01:
                    # Error increased, break
                    break
                prev_err = err
                
                _update_W(U, V, W, n, m, p, target_rank)
                _update_U(U, V, W, n, m, p, target_rank)
                _update_V(U, V, W, n, m, p, target_rank)
                
                if found:
                    break
            
            if found:
                break
        
        if found:
            continue  # Try to find even lower rank
    
    # If we found a better solution, return it
    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Otherwise, return the schoolbook solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
