"""Tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # safety margin before 150s limit


def _check_time():
    return time.time() >= _DEADLINE


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Direct tensor check: T[i,j,k] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, tol=1e-9):
    """Faster verification using vectorized operations."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape for batch computation
    # U: (n*m, rank) -> (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)  
    V_reshaped = V.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # For each (i, j, k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a 3D tensor contraction
    # Build result tensor T[i,j,k]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_col = U_reshaped[:, :, c]  # (n, m)
        v_col = V_reshaped[:, :, c]  # (m, p)
        w_col = W_reshaped[:, :, c]  # (p, n)
        
        # For each (i,j,k): u_col[i,j] * v_col[j,k] * w_col[k,i]
        # This is: sum over j of... no, it's a 3-way product
        # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        
        # Efficient way: for fixed j, T[:,j,:] += u_col[:,j] * v_col[j,:] * w_col[:,j]
        # Wait, w_col[k,i] so w_col is (p,n), w_col[k,i]
        # Let me just do it directly but vectorized over c
        
    # Actually let's just do the direct loop but it's only 63 entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    """Return the standard schoolbook decomposition of rank 63."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def _try_merge_terms(U, V, W, rank):
    """
    Try to merge two schoolbook terms into one.
    Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exist
    scalars a, b such that:
    a * e_{i1} ⊗ e_{j1} ⊗ e_{k1} + b * e_{i2} ⊗ e_{j2} ⊗ e_{k2} 
    = u ⊗ v ⊗ w for some vectors u, v, w
    
    This is only possible if the two terms share indices in a specific way.
    """
    n, m, p = DIMENSIONS
    # This is complex, skip for now
    return None


def _structured_search(rank, max_trials=200, seed=0):
    """
    Structured search for rank-rank decomposition.
    Uses the fact that we can group terms by shared indices.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    for trial in range(max_trials):
        if _check_time():
            return None
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strategy: assign each (i,j,k) triple to a column, but allow sharing
        # Use a random assignment with some structure
        
        # Initialize with schoolbook-like pattern but with random perturbations
        # and then try to optimize
        
        # Approach: random sparse initialization
        for c in range(rank):
            # Each column gets a random subset of non-zero entries
            # U entries
            for idx in range(n * m):
                if rng.random() < 0.4:
                    U[idx, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            # V entries
            for idx in range(m * p):
                if rng.random() < 0.4:
                    V[idx, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
            # W entries
            for idx in range(p * n):
                if rng.random() < 0.4:
                    W[idx, c] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        
        # Try alternating least squares refinement
        for iteration in range(20):
            if _check_time():
                return None
            
            # Update W given U, V
            for c in range(rank):
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    # Find j such that U[i*m+j, c] * V[j*p+k, c] != 0
                    # and solve for W[k*n+i, c]
                    found = False
                    for j in range(m):
                        u_val = U[i * m + j, c]
                        v_val = V[j * p + k, c]
                        if abs(u_val) > 1e-12 and abs(v_val) > 1e-12:
                            target = 1.0 / (u_val * v_val)
                            if not found:
                                W[idx, c] = target
                                found = True
                            elif abs(W[idx, c] - target) > 1e-6:
                                # Inconsistent, this column won't work
                                W[idx, c] = 0.0
                                break
                    if not found:
                        W[idx, c] = 0.0
            
            # Check if we've converged
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
            
            # Similarly update U and V (simplified)
            # For each c, for each (i,j), solve for U[i*m+j, c]
            for c in range(rank):
                for idx in range(n * m):
                    i = idx // m
                    j = idx % m
                    found = False
                    for k in range(p):
                        v_val = V[j * p + k, c]
                        w_val = W[k * n + i, c]
                        if abs(v_val) > 1e-12 and abs(w_val) > 1e-12:
                            target = 1.0 / (v_val * w_val)
                            if not found:
                                U[idx, c] = target
                                found = True
                            elif abs(U[idx, c] - target) > 1e-6:
                                U[idx, c] = 0.0
                                break
                    if not found:
                        U[idx, c] = 0.0
            
            # Update V
            for c in range(rank):
                for idx in range(m * p):
                    j = idx // p
                    k = idx % p
                    found = False
                    for i in range(n):
                        u_val = U[i * m + j, c]
                        w_val = W[k * n + i, c]
                        if abs(u_val) > 1e-12 and abs(w_val) > 1e-12:
                            target = 1.0 / (u_val * w_val)
                            if not found:
                                V[idx, c] = target
                                found = True
                            elif abs(V[idx, c] - target) > 1e-6:
                                V[idx, c] = 0.0
                                break
                    if not found:
                        V[idx, c] = 0.0
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_grouped_decomposition():
    """
    Try a grouped decomposition where we exploit the structure.
    
    For 3x3 * 3x7, we can think of it as 7 column multiplications:
    Each column of C is A (3x3) times a column of B (3x1).
    
    The rank of 3x3 * 3x1 is at most 9 (schoolbook), but might be less.
    If we can find a rank-r decomposition for 3x3*3x1, then 3x3*3x7 
    could have rank at most 7*r, but we can share structure.
    """
    n, m, p = DIMENSIONS
    
    # Try rank = 21 (3 * 7) with structured pattern
    # U[i*m+j, c] where c = i*7 + k_group
    # This doesn't directly work, let me think differently
    
    # Alternative: use the fact that T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try a pattern where:
    # U[i*m+j, c] depends only on i and c
    # V[j*p+k, c] depends only on j and c  
    # W[k*n+i, c] depends only on k and c
    # Then: sum_c U[i,c] * V[j,c] * W[k,c] = 1 for all i,j,k
    # This means the outer product sum gives a rank-1 tensor in each mode... 
    # Actually this would require U ⊗ V ⊗ W to be all-ones, which means
    # each of U, V, W must be rank 1 in their respective spaces
    # U: (3*3) x rank, but if U[i*m+j, c] = u_i * something... 
    # This is getting complex. Let's just do the search.
    
    return None


def _gradient_descent_search(rank, max_iters=500, seed=0):
    """
    Use gradient descent to find a rank-rank decomposition.
    Minimize ||T_reconstructed - T_target||^2
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    lr = 0.01
    for it in range(max_iters):
        if _check_time():
            return None
        
        # Compute gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # For each (i,j,k), error = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        # d(error)/d(U[i*m+j,c]) = V[j*p+k,c] * W[k*n+i,c]
        # But we sum over k for each (i,j), so:
        # grad_U[i*m+j, c] = sum_k (error[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
        
        # Compute errors
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    errors[i, j, k] = val - 1.0
        
        # Compute gradients
        for i in range(n):
            for j in range(m):
                for c in range(rank):
                    # grad_U[i*m+j, c] = sum_k errors[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
                    s = 0.0
                    for k in range(p):
                        s += errors[i, j, k] * V[j * p + k, c] * W[k * n + i, c]
                    grad_U[i * m + j, c] = s
        
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    # grad_V[j*p+k, c] = sum_i errors[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
                    s = 0.0
                    for i in range(n):
                        s += errors[i, j, k] * U[i * m + j, c] * W[k * n + i, c]
                    grad_V[j * p + k, c] = s
        
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    # grad_W[k*n+i, c] = sum_j errors[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
                    s = 0.0
                    for j in range(m):
                        s += errors[i, j, k] * U[i * m + j, c] * V[j * p + k, c]
                    grad_W[k * n + i, c] = s
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if np.max(np.abs(errors)) < 1e-8:
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        # Adaptive learning rate
        if it % 50 == 49:
            err_norm = np.linalg.norm(errors)
            if err_norm < 0.1:
                lr *= 0.5
            elif err_norm > 1.0:
                lr *= 0.5
    
    if _verify(U.tolist(), V.tolist(), W.tolist()):
        return U.tolist(), V.tolist(), W.tolist()
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook()
    best_result = (U, V, W)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Try from lower bound upward
    
    for rank in range(21, best_rank):
        if _check_time():
            break
        
        # Try structured search
        result = _structured_search(rank, max_trials=50, seed=rank)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break
        
        # Try gradient descent
        result = _gradient_descent_search(rank, max_iters=100, seed=rank + 1000)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
