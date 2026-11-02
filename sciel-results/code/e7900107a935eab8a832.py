"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for arr in [U, V, W]:
        if len(arr) == 0:
            return False
        if len(arr[0]) != rank:
            return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _tensor_values(n, m, p):
    """Get the target tensor values T[i,j,k] for matrix multiplication."""
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1.0
    return T

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual error for a decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(total - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _local_search(U, V, W, n, m, p, max_iters=500, tol=1e-10):
    """Perform local search to reduce residual."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for it in range(max_iters):
        # Try updating each entry of U, V, W to reduce residual
        improved = False
        
        # Update U entries
        for idx in range(n * m):
            for c in range(rank):
                old_val = U_arr[idx, c]
                # Compute current residual contribution
                # For each (i,j,k) where U[idx,c] appears:
                # idx = i*m + j, so i = idx // m, j = idx % m
                i = idx // m
                j = idx % m
                
                best_val = old_val
                best_err = float('inf')
                
                # Try a few candidate values
                for new_val in [0.0, 1.0, -1.0, 0.5, -0.5, old_val * 0.5, old_val * 2.0]:
                    U_arr[idx, c] = new_val
                    err = 0.0
                    for k in range(p):
                        total = 0.0
                        for cc in range(rank):
                            total += U_arr[i*m+j, cc] * V_arr[j*p+k, cc] * W_arr[k*n+i, cc]
                        err += (total - 1.0) ** 2
                    if err < best_err:
                        best_err = err
                        best_val = new_val
                U_arr[idx, c] = best_val
                if best_val != old_val:
                    improved = True
        
        # Update V entries
        for idx in range(m * p):
            for c in range(rank):
                old_val = V_arr[idx, c]
                j = idx // p
                k = idx % p
                
                best_val = old_val
                best_err = float('inf')
                
                for new_val in [0.0, 1.0, -1.0, 0.5, -0.5, old_val * 0.5, old_val * 2.0]:
                    V_arr[idx, c] = new_val
                    err = 0.0
                    for i in range(n):
                        total = 0.0
                        for cc in range(rank):
                            total += U_arr[i*m+j, cc] * V_arr[j*p+k, cc] * W_arr[k*n+i, cc]
                        err += (total - 1.0) ** 2
                    if err < best_err:
                        best_err = err
                        best_val = new_val
                V_arr[idx, c] = best_val
                if best_val != old_val:
                    improved = True
        
        # Update W entries
        for idx in range(p * n):
            for c in range(rank):
                old_val = W_arr[idx, c]
                k = idx // n
                i = idx % n
                
                best_val = old_val
                best_err = float('inf')
                
                for new_val in [0.0, 1.0, -1.0, 0.5, -0.5, old_val * 0.5, old_val * 2.0]:
                    W_arr[idx, c] = new_val
                    err = 0.0
                    for j in range(m):
                        total = 0.0
                        for cc in range(rank):
                            total += U_arr[i*m+j, cc] * V_arr[j*p+k, cc] * W_arr[k*n+i, cc]
                        err += (total - 1.0) ** 2
                    if err < best_err:
                        best_err = err
                        best_val = new_val
                W_arr[idx, c] = best_val
                if best_val != old_val:
                    improved = True
        
        if not improved:
            break
        
        # Check convergence
        max_err = _compute_residual(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), n, m, p)
        if max_err < tol:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def _random_init(n, m, p, rank):
    """Generate a random initialization."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        # Random sparse vectors
        n_nonzero = random.randint(1, 3)
        U_nonzero = random.sample(range(n * m), min(n_nonzero, n * m))
        for idx in U_nonzero:
            U[idx, c] = random.choice([1.0, -1.0, 0.5, -0.5])
        
        V_nonzero = random.sample(range(m * p), min(n_nonzero, m * p))
        for idx in V_nonzero:
            V[idx, c] = random.choice([1.0, -1.0, 0.5, -0.5])
        
        W_nonzero = random.sample(range(p * n), min(n_nonzero, p * n))
        for idx in W_nonzero:
            W[idx, c] = random.choice([1.0, -1.0, 0.5, -0.5])
    
    return U.tolist(), V.tolist(), W.tolist()

def _structured_init(n, m, p, rank):
    """Generate a structured initialization based on schoolbook with merging."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Start with schoolbook pattern but randomly merge some terms
    # Each term covers a subset of (i,j,k) indices
    indices = list(product(range(n), range(m), range(p)))
    random.shuffle(indices)
    
    for c in range(rank):
        # Assign 1-3 (i,j,k) tuples to this term
        num_assigned = random.randint(1, min(3, len(indices)))
        for _ in range(num_assigned):
            if not indices:
                break
            i, j, k = indices.pop()
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_greedy_rank(n, m, p, target_rank, timeout=30):
    """Try to find a decomposition of given rank using greedy approach."""
    start = time.time()
    deadline = start + timeout
    
    best_result = None
    best_err = float('inf')
    
    num_attempts = 20
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Choose initialization strategy
        if attempt % 2 == 0:
            U, V, W = _random_init(n, m, p, target_rank)
        else:
            U, V, W = _structured_init(n, m, p, target_rank)
        
        # Local search
        U, V, W = _local_search(U, V, W, n, m, p, max_iters=200)
        
        # Check result
        max_err = _compute_residual(U, V, W, n, m, p)
        if max_err < 1e-9:
            return {"U": U, "V": V, "W": W}
        
        if max_err < best_err:
            best_err = max_err
            best_result = {"U": U, "V": V, "W": W}
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    deadline = time.time() + 110  # Leave margin before 150s limit
    
    # Try to find lower-rank decompositions
    # Lower bound: max(m*p, n*p, n*m) = max(28, 21, 12) = 28
    # But realistic lower bound is higher
    
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 5:
            break
        
        search_time = min(10, deadline - time.time() - 2)
        if search_time < 2:
            break
        
        result_rank = _try_greedy_rank(n, m, p, target_rank, timeout=search_time)
        if result_rank is not None:
            result = result_rank
            best_rank = target_rank
            # Continue searching for even lower ranks
    
    return result

# EVOLVE-BLOCK-END
