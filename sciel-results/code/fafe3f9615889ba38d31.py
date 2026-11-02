"""Matrix multiplication 3x4x5 - explicit rank-minimal decomposition attempt."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[x,y,z] where:
    x = i*m+j (A index), y = j*p+k (B index), z = k*n+i (C index, column-major)
    T[x,y,z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i for some consistent (i,j,k), else 0.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _compute_decomposed_tensor(U, V, W):
    """Compute D[x,y,z] = sum_r U[x,r] * V[y,r] * W[z,r]"""
    return np.einsum('xr,yr,zr->xyz', U, V, W)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    expected = _build_expected_tensor(n, m, p)
    D = _compute_decomposed_tensor(U, V, W)
    return np.allclose(D, expected, atol=tol)

def _try_rank_random(n, m, p, rank, num_trials=50, time_limit=2.0, seed=None):
    """Try to find a valid rank-r decomposition via random search.
    
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    T = _build_expected_tensor(n, m, p)
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with structured patterns
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        
        # Try different normalization strategies
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-12 and norm_v > 1e-12 and norm_w > 1e-12:
                # Scale to unit norm
                U[:, r] /= norm_u
                V[:, r] /= norm_v
                W[:, r] /= norm_w
        
        # Check if already close
        D = _compute_decomposed_tensor(U, V, W)
        err = np.max(np.abs(D - T))
        if err < 1e-9:
            return U, V, W
    
    return None

def _try_rank_structured(n, m, p, rank, time_limit=2.0, seed=None):
    """Try structured decompositions with specific patterns.
    
    This explores structured families that might achieve lower rank.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    T = _build_expected_tensor(n, m, p)
    
    # Strategy 1: Block-structured decomposition
    # For 3x4x5, try to exploit the fact that 4 = 2*2
    
    # Try rank = n*p*m - reduction
    # Known lower bound for matrix multiplication is max(n*p, m*p, n*m) = 15
    # But actual rank is likely higher
    
    # Attempt structured patterns:
    # Pattern A: Diagonal sharing
    # Pattern B: Kronecker-like structure
    
    # For now, try a few specific structured initializations
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts and time.time() - start < time_limit:
        attempts += 1
        
        # Random structured initialization
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Fill with random sparse patterns
        for r in range(rank):
            # Each rank term is a rank-1 tensor
            # Choose random support patterns
            u_sparsity = rng.random(n*m) < 0.3
            v_sparsity = rng.random(m*p) < 0.3
            w_sparsity = rng.random(p*n) < 0.3
            
            U[u_sparsity, r] = rng.standard_normal(u_sparsity.sum())
            V[v_sparsity, r] = rng.standard_normal(v_sparsity.sum())
            W[w_sparsity, r] = rng.standard_normal(w_sparsity.sum())
        
        # Verify
        D = _compute_decomposed_tensor(U, V, W)
        err = np.max(np.abs(D - T))
        if err < 1e-9:
            return U, V, W
    
    return None

def _try_rank_alternating(n, m, p, rank, num_iters=100, time_limit=2.0, seed=None):
    """Try alternating minimization to refine a decomposition.
    
    This is a heuristic approach that might find valid decompositions.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    T = _build_expected_tensor(n, m, p)
    
    # Initialize
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))
    
    # Normalize
    for r in range(rank):
        norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
        if norm > 1e-12:
            scale = norm ** (1.0/3.0)
            U[:,r] /= scale
            V[:,r] /= scale
            W[:,r] /= scale
    
    # Alternating minimization
    for it in range(num_iters):
        if time.time() - start > time_limit:
            break
        
        # Check current error
        D = _compute_decomposed_tensor(U, V, W)
        err = np.max(np.abs(D - T))
        if err < 1e-9:
            return U, V, W
        
        # Update U: for each x, solve min_Ux sum_yz (T[x,y,z] - sum_r' U[x,r'] V[y,r'] W[z,r'])^2
        # This is a linear system
        # Simplified: just do a small step gradient descent
        lr = 0.01
        
        # Compute gradient w.r.t U
        # grad_U[x,r] = sum_yz (D[x,y,z] - T[x,y,z]) * V[y,r] * W[z,r]
        residual = D - T
        grad_U = np.einsum('xyz,yr,zr->xr', residual, V, W)
        U -= lr * grad_U
        
        # Recompute D
        D = _compute_decomposed_tensor(U, V, W)
        residual = D - T
        grad_V = np.einsum('xyz,xr,zr->yr', residual, U, W)
        V -= lr * grad_V
        
        D = _compute_decomposed_tensor(U, V, W)
        residual = D - T
        grad_W = np.einsum('xyz,xr,yr->zr', residual, U, V)
        W -= lr * grad_W
        
        # Renormalize
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
            if norm > 1e-12:
                scale = norm ** (1.0/3.0)
                U[:,r] /= scale
                V[:,r] /= scale
                W[:,r] /= scale
    
    # Final check
    D = _compute_decomposed_tensor(U, V, W)
    err = np.max(np.abs(D - T))
    if err < 1e-9:
        return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    # Verify schoolbook (sanity check)
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        # Should never happen, but return schoolbook anyway
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find lower-rank decompositions
    best_result = (schoolbook_rank, U_school, V_school, W_school)
    
    # Try ranks from schoolbook-1 down, but limit total time
    remaining_time = deadline - time.time()
    
    # Try a few specific ranks with different strategies
    ranks_to_try = [59, 58, 55, 50, 45]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:  # 10s safety margin
            break
        
        time_per_rank = max(1.0, (deadline - time.time() - 10) / max(1, len(ranks_to_try)))
        
        # Try random search
        result = _try_rank_random(n, m, p, rank, num_trials=20, time_limit=time_per_rank/3, seed=rank)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = (rank, U, V, W)
                break
        
        # Try structured search
        if time.time() < deadline - 10:
            result = _try_rank_structured(n, m, p, rank, time_limit=time_per_rank/3, seed=rank+100)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    best_result = (rank, U, V, W)
                    break
        
        # Try alternating minimization
        if time.time() < deadline - 10:
            result = _try_rank_alternating(n, m, p, rank, num_iters=50, time_limit=time_per_rank/3, seed=rank+200)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    best_result = (rank, U, V, W)
                    break
    
    rank, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
