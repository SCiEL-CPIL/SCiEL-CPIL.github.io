"""Matrix multiplication 3x3 x 3x7 rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# Known rank for 3x3 * 3x7: 
# This is equivalent to 7 independent 3x3 * 3x1 (matrix-vector) products.
# The rank of a single 3x3 * 3x1 product is 9 (schoolbook) but can be reduced.
# Actually, the tensor for 3x3 * 3x7 is T_3,3,7.
# The trivial rank is 3*3*7 = 63.
# The lower bound is max(3*3, 3*7, 7*3) = 21.
# 
# Let's check if we can use Strassen-like algorithms or known results.
# For 2x2 * 2x2, rank is 7.
# For 3x3 * 3x3, rank is at most 23 (Strassen) or better.
# For 3x3 * 3x7, we can view it as multiplying A (3x3) by B (3x7).
# B has 7 columns. We can compute A * b_i for each column b_i.
# If we had an algorithm for 3x3 * 3x1 with rank r, we could use 7*r.
# But we can do better by batching.
#
# Actually, there is a known result: rank(3,3,7) <= 42? Or maybe lower.
# Let's try to find a decomposition with rank significantly lower than 63.
# 
# Strategy: Use a structured search or known algorithmic decompositions.
# Since 3 is small, we might be able to find a low-rank decomposition by exploiting symmetries or using known tensor decompositions.
#
# One approach: The tensor T_3,3,7 can be seen as a block diagonal of 7 copies of T_3,3,1? No, that's not right.
# T_3,3,7[i, j, k] = 1 if i in [0,2], j in [0,2], k in [0,6].
# Indices: a = 3i + j, b = 7j + k, c = 3k + i.
#
# Let's try a random search with a better initialization or a gradient-based method with more iterations.
# Or, we can try to find a decomposition with rank 21 (the lower bound) or close to it.
#
# Given the time constraint, I'll improve the random search to be more effective.
# I will also try a few specific ranks like 21, 22, ..., 30.
#
# To make the search more effective, I will:
# 1. Use a better initialization (e.g., based on SVD of random projections).
# 2. Use a more robust optimization method (e.g., Adam or RMSProp).
# 3. Try multiple restarts for each rank.

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    Tensor shape: (n*m, m*p, p*n) in the flattened index convention.
    T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    This is the standard matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check if U @ diag @ V^T @ diag @ W^T reconstructs T.
    
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Reconstructed: sum_c U[:,c] * V[:,c] * W[:,c] = T
    """
    rank = U.shape[1]
    # Compute tensor contraction: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Efficient: for each r, outer product
    T_recon = np.zeros(T.shape, dtype=np.float64)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol, rtol=tol)

def _verify_decomposition_fast(U, V, W, T, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Reshape: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # For each r, we have rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r
    
    # Alternative: use einsum for the whole thing
    # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=tol)

def _try_random_rank(rank, T, max_iterations=500, time_limit=10.0):
    """Try to find a rank-`rank` decomposition using random initialization and gradient descent with momentum."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    # Use a more effective optimizer: Adam
    # Initialize parameters
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Momentum buffers
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        lr = 0.01
        
        for sub_it in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_recon - T)
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                if error < 1e-9:
                    return U.copy(), V.copy(), W.copy(), True
            
            # Compute gradients
            residual = T - T_recon
            grad_U = -2 * np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = -2 * np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = -2 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            # Adam update
            t = sub_it + 1
            mU = b1 * mU + (1 - b1) * grad_U
            mV = b1 * mV + (1 - b1) * grad_V
            mW = b1 * mW + (1 - b1) * grad_W
            vU = b2 * vU + (1 - b2) * (grad_U ** 2)
            vV = b2 * vV + (1 - b2) * (grad_V ** 2)
            vW = b2 * vW + (1 - b2) * (grad_W ** 2)
            
            mU_hat = mU / (1 - b1 ** t)
            mV_hat = mV / (1 - b1 ** t)
            mW_hat = mW / (1 - b1 ** t)
            vU_hat = vU / (1 - b2 ** t)
            vV_hat = vV / (1 - b2 ** t)
            vW_hat = vW / (1 - b2 ** t)
            
            U += lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V += lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W += lr * mW_hat / (np.sqrt(vW_hat) + eps)
    
    # Check best
    if best_U is not None:
        T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
        error = np.linalg.norm(T_recon - T)
        if error < 1e-9:
            return best_U, best_V, best_W, True
    
    return None, None, None, False

def _schoolbook():
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _to_list_of_lists(arr):
    """Convert numpy array to list of lists of floats (or ints if close to integers)."""
    return arr.tolist()

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Start with schoolbook as fallback
    fallback = _schoolbook()
    
    # Time budget management
    start_time = time.time()
    total_budget = 120.0
    margin = 5.0
    effective_budget = total_budget - margin
    
    # Try to find better decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Known: rank of 3x3 * 3x7 is between 21 and 63
    # Let's try various ranks from the lower bound up
    
    # First, try a structured approach: exploit that one dimension is small
    # For A(3x3) * B(3x7), we can think of this as computing 3 rows of A times 7 columns of B
    
    # Attempt: use the fact that 3x3 multiplication has rank 7 (Strassen-like)
    # But we're multiplying 3x3 by 3x7, so it's like 7 separate 3x3 * 3x1 multiplications
    # Each 3x3 * 3x1 (matrix-vector) has rank 9 in general, but can be lower
    
    # Let's try a more systematic search with time limits
    T = _build_tensor()
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)  # 21
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try a few candidate ranks
    # Focus on ranks that are likely to be achievable
    candidate_ranks = list(range(lower_bound, min(lower_bound + 10, schoolbook_rank)))
    
    for rank in candidate_ranks:
        if time.time() - start_time > effective_budget:
            break
        
        # Time per rank attempt
        time_per_rank = min(20.0, (effective_budget - (time.time() - start_time)) / max(1, len(candidate_ranks)))
        
        result = _try_random_rank(rank, T, max_iterations=200, time_limit=time_per_rank)
        if result[3]:
            U, V, W, _ = result
            best_rank = rank
            best_result = {"U": _to_list_of_lists(U), "V": _to_list_of_lists(V), "W": _to_list_of_lists(W)}
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_result is not None:
        return best_result
    
    return fallback
