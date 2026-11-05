"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Check all valid entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(s - 1.0) > tol:
                    return False
    
    # Check that invalid entries are zero
    valid_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_entries.add((i*m+j, j*p+k, k*n+i))
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid_entries:
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _als_step(U, V, W, T, n, m, p, target_rank, deadline):
    """One ALS iteration for tensor decomposition."""
    if time.time() > deadline:
        return U, V, W, None
    
    # Solve for W: fix U, V
    # For each c: T[:, :, c].ravel() = sum_r W[c,r] * (U[:,r] outer V[:,r]).ravel()
    W_new = np.zeros_like(W)
    for c in range(p * n):
        M = np.zeros((n*m * m*p, target_rank))
        for r in range(target_rank):
            M[:, r] = np.outer(U[:, r], V[:, r]).ravel()
        rhs = T[:, :, c].ravel()
        sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        W_new[c, :] = sol
    W = W_new
    
    # Solve for U: fix V, W
    U_new = np.zeros_like(U)
    for a in range(n * m):
        M = np.zeros((m*p * p*n, target_rank))
        for r in range(target_rank):
            M[:, r] = np.outer(V[:, r], W[:, r]).ravel()
        rhs = T[a, :, :].ravel()
        sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        U_new[a, :] = sol
    U = U_new
    
    # Solve for V: fix U, W
    V_new = np.zeros_like(V)
    for b in range(m * p):
        M = np.zeros((n*m * p*n, target_rank))
        for r in range(target_rank):
            M[:, r] = np.outer(U[:, r], W[:, r]).ravel()
        rhs = T[:, b, :].ravel()
        sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        V_new[b, :] = sol
    V = V_new
    
    # Compute error
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                err += (s - 1.0)**2
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Skip valid entries (already counted)
                i = a // m
                j = a % m
                # Check if this is a valid entry
                is_valid = False
                for k in range(p):
                    if b == j*p + k and c == k*n + i:
                        is_valid = True
                        break
                if not is_valid:
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += s**2
    
    return U, V, W, err

def _search_rank(T, n, m, p, target_rank, deadline, max_restarts=8, max_iters=30):
    """Search for a rank-target_rank decomposition using ALS with multiple restarts."""
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(1000 + restart * 7)
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_step(U, V, W, T, n, m, p, target_rank, deadline)
            
            if err is not None:
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if err < 1e-12:
                        return best_U, best_V, best_W
    
    if best_U is not None and best_err < 1e-10:
        return best_U, best_V, best_W
    return None, None, None

def _to_half_integer(mat):
    """Convert to half-integers if close."""
    mat = np.array(mat, dtype=np.float64)
    rounded = np.round(mat * 2) / 2.0
    if np.allclose(mat, rounded, atol=1e-6):
        mat = rounded
    return mat

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook as safe baseline
    school = _schoolbook(n, m, p)
    best_U = school["U"]
    best_V = school["V"]
    best_W = school["W"]
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known results suggest rank can be lower than n*m*p
    # Try ranks in decreasing order from just below schoolbook
    candidate_ranks = [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
        
        U, V, W = _search_rank(T, n, m, p, rank, deadline, max_restarts=5, max_iters=25)
        
        if U is not None:
            # Verify the solution
            if _verify_exact(U, V, W, n, m, p, tol=1e-10):
                best_U = U
                best_V = V
                best_W = W
                best_rank = rank
                # Found a valid solution; can try to go lower
                continue
        else:
            # No valid solution found for this rank; try lower ranks
            continue
    
    # Convert to lists
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
