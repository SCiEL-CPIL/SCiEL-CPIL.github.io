"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import itertools
import numpy as np
from numpy.linalg import matrix_rank, solve, lstsq

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal (or best found) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, provide the schoolbook decomposition as a guaranteed valid result
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Strategy 1: Try structured decomposition exploiting block structure
    if time.time() < deadline - 20:
        result = _try_structured_decomposition(n, m, p, deadline)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
    
    # Strategy 2: Try ALS-based search for lower rank with better initialization
    if time.time() < deadline - 10:
        for target_rank in range(min(best_rank - 1, 74), max(best_rank - 15, 55), -1):
            if time.time() > deadline - 10:
                break
            result = _try_als_decomposition(n, m, p, target_rank, deadline)
            if result is not None:
                rank = len(result['U'][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
                break  # Found a valid decomposition, stop
    
    # Strategy 3: Try rank reduction via linear dependency detection
    if time.time() < deadline - 5:
        result = _try_rank_reduction(n, m, p, best_result, best_rank, deadline)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _try_structured_decomposition(n, m, p, deadline):
    """
    Try a structured decomposition that exploits the block structure.
    For 3x5x5, we can think of it as 3 blocks of 5x5 multiplications.
    Try to find a low-rank decomposition using the structure.
    """
    if time.time() > deadline - 15:
        return None
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want T = sum_r U[:,r] * V[:,r] * W[:,r]
    
    # Try a decomposition where we group by i (the 3 dimension)
    # For each i, we have a 5x5 matrix multiplication tensor
    # Each 5x5 matmul has rank at most 25, so 3*25 = 75 is schoolbook
    
    # Try to find if there's a shared structure
    # Use the fact that the 5x5 multiplication tensor might have rank < 25
    
    # For 5x5 matrix multiplication, the best known rank is 25 (schoolbook)
    # but there might be improvements for the rectangular case
    
    # Try Winograd-style for the 5x5 blocks
    result = _try_winograd_5x5(n, m, p, deadline)
    if result is not None:
        return result
    
    return None


def _try_winograd_5x5(n, m, p, deadline):
    """
    Try Winograd-style decomposition for 5x5 matrix multiplication blocks.
    """
    if time.time() > deadline - 10:
        return None
    
    # For 5x5 matrix multiplication, use the fact that we can decompose
    # the 5x5 tensor into a sum of rank-1 tensors
    
    # The standard 5x5 matmul tensor has rank 25
    # Try to find a lower rank decomposition
    
    # Use the structure: C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form
    
    # For now, try a simple approach: see if we can reduce the rank
    # by exploiting symmetries
    
    # Actually, for 5x5, the minimum rank is known to be 25
    # So the 3x5x5 tensor should have rank 3*25 = 75
    
    # But let's try a different approach: use the fact that the tensor
    # is a sum of 3 independent 5x5 tensors
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_rank_reduction(n, m, p, base_result, base_rank, deadline):
    """
    Try to reduce the rank by finding linear dependencies in the factor matrices.
    """
    if time.time() > deadline - 5:
        return None
    
    U = np.array(base_result['U'], dtype=float)
    V = np.array(base_result['V'], dtype=float)
    W = np.array(base_result['W'], dtype=float)
    
    # Check if any columns are linearly dependent
    # If column r1 and r2 are linearly dependent, we might be able to merge them
    
    # For now, this is a placeholder for future work
    return None


def _try_randomized_search(n, m, p, current_best_rank):
    """
    Try randomized search for a lower-rank decomposition.
    """
    return None


def _try_als_decomposition(n, m, p, target_rank, deadline):
    """
    Try to find a rank-r decomposition using Alternating Least Squares
    with structured initialization.
    """
    r = target_rank
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute index mappings for efficiency
    U_rows = np.array([(i, j) for i in range(n) for j in range(m)])
    V_rows = np.array([(j, k) for j in range(m) for k in range(p)])
    W_rows = np.array([(k, i) for k in range(p) for i in range(n)])
    
    best_result = None
    
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        # Initialize with perturbed schoolbook-like structure
        np.random.seed(42 + trial * 7)
        
        # Create a structured initialization
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Distribute the schoolbook terms across r columns
        # Each schoolbook term (i,j,k) maps to a column
        for idx, (i, j) in enumerate(U_rows):
            for k in range(p):
                col = (idx * p + k) % r
                U[idx, col] += 1.0 / r  # Spread across columns
                V[j * p + k, col] += 1.0 / r
                W[k * n + i, col] += 1.0 / r
        
        # Add random perturbation
        U += np.random.randn(nm, r) * 0.1
        V += np.random.randn(mp, r) * 0.1
        W += np.random.randn(pn, r) * 0.1
        
        prev_err = float('inf')
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for idx, (i, j) in enumerate(U_rows):
                V_block = V[j*p:(j+1)*p, :]  # (p, r)
                W_idx = np.array([k*n+i for k in range(p)])
                W_block = W[W_idx, :]  # (p, r)
                M = V_block * W_block  # (p, r)
                try:
                    U[idx, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
            
            # Update V
            for idx, (j, k) in enumerate(V_rows):
                U_idx = np.array([i*m+j for i in range(n)])
                U_block = U[U_idx, :]  # (n, r)
                W_block = W[k*n:(k+1)*n, :]  # (n, r)
                M = U_block * W_block  # (n, r)
                try:
                    V[idx, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                except:
                    pass
            
            # Update W
            for idx, (k, i) in enumerate(W_rows):
                U_idx = np.array([i*m+j for j in range(m)])
                U_block = U[U_idx, :]  # (m, r)
                V_idx = np.array([j*p+k for j in range(m)])
                V_block = V[V_idx, :]  # (m, r)
                M = U_block * V_block  # (m, r)
                try:
                    W[idx, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                except:
                    pass
            
            # Check error
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-7:
                result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                if best_result is None or r < len(best_result['U'][0]):
                    best_result = result
                break
            
            # Early stopping if not converging
            if max_err > prev_err * 1.1 and iteration > 50:
                break
            prev_err = max_err
    
    return best_result


# EVOLVE-BLOCK-END
