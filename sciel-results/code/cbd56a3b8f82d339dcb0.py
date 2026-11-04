"""Tensor decomposition search for 4x4x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

def _tensor_values(U, V, W, n, m, p):
    """Compute the tensor values from factor matrices."""
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                tensor[i, j, k] = val
    return tensor

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly matches the matrix multiplication tensor."""
    tensor = _tensor_values(U, V, W, n, m, p)
    # The target tensor for matrix multiplication: T[i,j,k] = 1 if this is a valid (i,j,k) triple
    # Actually, the multiplication tensor T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} = 1 for all valid indices
    # Wait, that's not right. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that each term contributes)
    # Actually, the tensor is defined such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # So we need tensor[i,j,k] = 1 for all i,j,k
    target = np.ones((n, m, p))
    return np.allclose(tensor, target, atol=tol, rtol=0)

def _rank(U, V, W):
    """Return the rank (number of columns) of the decomposition."""
    return U.shape[1]

def _structural_lower_bound(n, m, p):
    """Structural lower bound on the rank of matrix multiplication tensor."""
    return max(n * m, m * p, p * n)

def _schoolbook(n, m, p):
    """Construct the schoolbook decomposition of rank n*m*p."""
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

def _greedy_build(n, m, p, max_rank, time_limit, rng):
    """
    Greedily build a decomposition by adding the best rank-1 term at each step.
    Returns (U, V, W) if a valid decomposition is found within max_rank, else None.
    """
    target = np.ones((n, m, p))
    residual = target.copy()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    start_time = time.time()
    
    for rank_so_far in range(max_rank):
        if time.time() - start_time > time_limit:
            return None
        
        # Find the best rank-1 term to add
        # A rank-1 term is u ⊗ v ⊗ w, which contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k]
        # We want to maximize the reduction in ||residual||_F
        # This is equivalent to finding u,v,w that maximize sum_{i,j,k} residual[i,j,k] * u[i,j] * v[j,k] * w[k,i]
        
        # This is a non-convex optimization. Use a simple heuristic:
        # Randomly sample candidate u, v, w and pick the best one.
        
        best_score = -np.inf
        best_u = None
        best_v = None
        best_w = None
        
        # Try random candidates
        num_samples = 200
        for _ in range(num_samples):
            # Sample u, v, w from {-1, 0, 1}
            u = rng.choice([-1, 0, 1], size=(n * m,))
            v = rng.choice([-1, 0, 1], size=(m * p,))
            w = rng.choice([-1, 0, 1], size=(p * n,))
            
            # Compute the contribution
            contrib = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        contrib[i, j, k] = u[i * m + j] * v[j * p + k] * w[k * n + i]
            
            # Score: how well does this reduce the residual?
            # We want to minimize ||residual - contrib||_F^2
            # = ||residual||_F^2 - 2 * sum(residual * contrib) + ||contrib||_F^2
            # So we want to maximize sum(residual * contrib) - 0.5 * ||contrib||_F^2
            score = np.sum(residual * contrib) - 0.5 * np.sum(contrib ** 2)
            
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_u is None:
            return None
        
        # Add this term
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        # Update residual
        contrib = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i, j, k] = best_u[i * m + j] * best_v[j * p + k] * best_w[k * n + i]
        residual -= contrib
        
        # Check if we've reached the target
        if np.allclose(residual, 0, atol=1e-9):
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
            return U, V, W
    
    return None

def _drop_terms(U, V, W, n, m, p, tol=1e-9):
    """
    Try to drop redundant terms from the decomposition.
    Returns (U, V, W) with some terms potentially removed.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Try dropping each term one at a time
    for r in range(rank):
        # Create copies without term r
        U_new = np.delete(U, r, axis=1)
        V_new = np.delete(V, r, axis=1)
        W_new = np.delete(W, r, axis=1)
        
        # Check if the decomposition is still valid
        if _is_valid(U_new, V_new, W_new, n, m, p, tol):
            U, V, W = U_new, V_new, W_new
            rank = U.shape[1]
            # Restart the loop since indices changed
            r = 0
            if rank <= 1:
                break
    
    return U, V, W

def _random_greedy(n, m, p, max_rank, time_limit, rng):
    """
    Randomized greedy construction with multiple restarts.
    Returns the best (U, V, W) found.
    """
    best_U = None
    best_V = None
    best_W = None
    best_rank = np.inf
    
    start_time = time.time()
    num_restarts = 0
    
    while time.time() - start_time < time_limit:
        num_restarts += 1
        result = _greedy_build(n, m, p, max_rank, time_limit - (time.time() - start_time), rng)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    if best_U is not None:
        return best_U, best_V, best_W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x5 matrix.
    
    Returns dict with U (n*m x rank), V (m*p x rank), W (p*n x rank)
    such that the multiplication tensor is exactly decomposed.
    
    Strategy:
    1. Construct the schoolbook decomposition (rank 80).
    2. Attempt to reduce rank using greedy construction with random restarts.
    3. Drop redundant terms from any valid decomposition found.
    4. If no improvement is found within the time limit, return the schoolbook.
    """
    t_start = time.time()
    deadline = t_start + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = U_school.shape[1]
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try greedy construction with various max_rank values
    rng = np.random.default_rng(42)
    
    # Try to find a decomposition with rank less than schoolbook
    for max_rank in [best_rank - 1, best_rank - 5, best_rank - 10, best_rank - 20]:
        if max_rank < _structural_lower_bound(n, m, p):
            continue
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        result = _random_greedy(n, m, p, max_rank, remaining_time - 2, rng)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
            if rank < best_rank:
                # Try to drop more terms
                U, V, W = _drop_terms(U, V, W, n, m, p)
                rank = U.shape[1]
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    # Also try to drop terms from the schoolbook
    U_drop, V_drop, W_drop = _drop_terms(U_school, V_school, W_school, n, m, p)
    if U_drop.shape[1] < best_rank:
        best_U, best_V, best_W = U_drop, V_drop, W_drop
        best_rank = U_drop.shape[1]
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
