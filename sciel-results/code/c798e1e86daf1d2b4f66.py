"""Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition.

We search for an exact decomposition of the tensor
    T[i,j,k] = 1  iff  (i,j,k) valid, with the output index mapped as W[k*n+i, r]
using integer coefficients.

The schoolbook rank is 45. We attempt to find a lower rank (44) using a
highly optimized local search with a strict time budget. If no lower rank
solution is found within the budget, we return the provably correct schoolbook
rank 45 decomposition.

Method:
  1. Construct the schoolbook rank-45 decomposition.
  2. Attempt local search for rank 44 with a 10-second wall-clock limit.
  3. Return the best valid solution found (likely rank 45).
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _compute_residual(U, V, W, n, m, p, rank):
    """
    Compute the residual tensor R where 
    R[i,j,k] = 1 - sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
    Returns a numpy array of shape (n, m, p).
    """
    # Initialize R with 1s
    R = np.ones((n, m, p), dtype=np.float64)
    
    # Iterate over rank terms
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        # Vectorized update for each (i, j)
        # u_val is scalar for fixed i, j
        # We can iterate i, j and update the slice R[i, j, :]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val != 0:
                    # v_slice is V[j*p : (j+1)*p, r]
                    v_slice = v_col[j * p:(j + 1) * p]
                    # w_slice is W[i::n, r] -> indices i, i+n, i+2n...
                    # W index is k*n + i. So for fixed i, k varies.
                    # W[:, r] has shape (p*n,).
                    # We need W[k*n + i, r] for k=0..p-1.
                    # This is W[i::n, r]
                    w_slice = w_col[i::n]
                    R[i, j, :] -= u_val * v_slice * w_slice
                    
    return R

def _verify_solution(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact (residual is 0)."""
    R = _compute_residual(U, V, W, n, m, p, rank)
    return np.max(np.abs(R)) < 1e-9

def _local_search_rank_r(n, m, p, rank, rng, deadline):
    """
    Local search for a rank-r decomposition.
    Returns (U, V, W) as numpy int arrays or None.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    # Initialize with random small integers
    # Using a small range increases the chance of finding integer solutions
    U = rng.integers(-2, 3, size=(N, rank))
    V = rng.integers(-2, 3, size=(P, rank))
    W = rng.integers(-2, 3, size=(Q, rank))
    
    max_iters = 2000
    step_size = 2
    
    for _iter in range(max_iters):
        if time.time() >= deadline:
            break
            
        R = _compute_residual(U, V, W, n, m, p, rank)
        err = np.abs(R)
        max_err = np.max(err)
        
        if max_err < 1e-9:
            return (U.astype(int), V.astype(int), W.astype(int))
        
        # Pick the index with the largest error
        ri, rj, rk = np.unravel_index(np.argmax(err), err.shape)
        target = 1.0
        current = 0.0
        for r in range(rank):
            current += U[ri * m + rj, r] * V[rj * p + rk, r] * W[rk * n + ri, r]
            
        deficit = target - current
        if abs(deficit) < 1e-12:
            continue

        # Try to adjust one entry in U, V, or W
        # Choose a random rank index r
        r = rng.integers(0, rank)
        
        uv = U[ri * m + rj, r]
        vv = V[rj * p + rk, r]
        ww = W[rk * n + ri, r]
        
        choice = rng.integers(0, 3)
        
        if choice == 0:
            if vv * ww == 0: continue
            delta = deficit / (vv * ww)
            new_val = uv + delta
            rounded = int(round(new_val))
            if abs(rounded - uv) <= step_size:
                U[ri * m + rj, r] = rounded
                # Check improvement
                new_current = 0.0
                for r2 in range(rank):
                    new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                if abs(new_current - target) < abs(current - target):
                    pass
                else:
                    U[ri * m + rj, r] = uv
        elif choice == 1:
            if uv * ww == 0: continue
            delta = deficit / (uv * ww)
            new_val = vv + delta
            rounded = int(round(new_val))
            if abs(rounded - vv) <= step_size:
                V[rj * p + rk, r] = rounded
                new_current = 0.0
                for r2 in range(rank):
                    new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                if abs(new_current - target) < abs(current - target):
                    pass
                else:
                    V[rj * p + rk, r] = vv
        else:
            if uv * vv == 0: continue
            delta = deficit / (uv * vv)
            new_val = ww + delta
            rounded = int(round(new_val))
            if abs(rounded - ww) <= step_size:
                W[rk * n + ri, r] = rounded
                new_current = 0.0
                for r2 in range(rank):
                    new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                if abs(new_current - target) < abs(current - target):
                    pass
                else:
                    W[rk * n + ri, r] = ww

    R = _compute_residual(U, V, W, n, m, p, rank)
    if np.max(np.abs(R)) < 1e-9:
        return (U.astype(int), V.astype(int), W.astype(int))
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Try to find rank 44 with a short time budget (10 seconds)
    # If found, return it. Otherwise, return schoolbook rank 45.
    deadline_search = start + 10.0
    rng = np.random.default_rng(42)
    
    # Attempt rank 44
    sol = _local_search_rank_r(n, m, p, 44, rng, deadline_search)
    if sol is not None:
        U, V, W = sol
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist(),
        }

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
