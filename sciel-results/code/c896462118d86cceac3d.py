"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import itertools
import numpy as np

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
    
    # Try to find better decompositions
    # For 3x5x5, the schoolbook rank is 75
    # Known lower bounds suggest rank could be lower
    
    # Strategy 1: Try Winograd-style transformations
    if time.time() < deadline:
        winograd_result = _try_winograd_rectangular(n, m, p)
        if winograd_result is not None:
            rank = len(winograd_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = winograd_result
    
    # Strategy 2: Try structured decompositions exploiting symmetry
    if time.time() < deadline:
        symmetric_result = _try_symmetric_decomposition(n, m, p)
        if symmetric_result is not None:
            rank = len(symmetric_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = symmetric_result
    
    # Strategy 3: Try rank reduction via linear combinations
    if time.time() < deadline:
        reduced_result = _try_rank_reduction(n, m, p, best_result, best_rank)
        if reduced_result is not None:
            rank = len(reduced_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = reduced_result
    
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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] appears
    in the (i,k)-th entry of the result.
    
    For each rank-1 term r:
      U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should sum to 1 
      when we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    More precisely, for each (i,k), the sum over j and r of
      U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 for the correct (j) terms and 0 otherwise.
    
    Actually, the correct verification is:
    For each (i, k), define the vector f_{i,k} where f_{i,k}[j] = 1.
    Then we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    to reconstruct the identity in the j-index.
    
    Let me reformulate: The matrix multiplication tensor has entries T[i,j,k] = 1
    for all valid i,j,k. A rank-r decomposition gives:
    T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right either. Let me think again.
    
    The matrix multiplication tensor M_{i,j,k} is defined as M_{i,j,k} = 1
    for all i in [0,n), j in [0,m), k in [0,p). This is the structure tensor.
    
    Actually no. The matrix multiplication tensor is:
    M_{i,j,k} = delta_{j,j} which is just 1 for all valid indices.
    
    Hmm, let me reconsider. In the standard formulation:
    (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor for this bilinear map has entries T_{i,k,j} = 1 (for all valid i,k,j).
    
    A rank-r decomposition means:
    T_{i,k,j} = sum_{r=1}^{r} U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
    
    Wait, the indices need to be consistent. Let me use the convention from the task:
    U has shape (n*m, rank) indexed by [i*m + j, r]
    V has shape (m*p, rank) indexed by [j*p + k, r]  
    W has shape (p*n, rank) indexed by [k*n + i, r]
    
    So the decomposition is:
    For each (i, j, k): 
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This must hold for all valid i, j, k.
    """
    rank = len(U[0])
    # Build numpy arrays for efficient verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over all rank terms
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_winograd_rectangular(n, m, p):
    """
    Try Winograd-style transformation for rectangular matrix multiplication.
    
    For n=3, m=5, p=5, we can try to exploit structure.
    Winograd for matrix multiplication typically works best for square matrices
    where the block size divides the dimension evenly.
    
    For 3x5x5, we can try:
    - Decompose the 5x5 part using Winograd with block size 2 or 5
    - Handle the 3x5 part separately
    
    This is a heuristic approach; we'll try small rank reductions.
    """
    # For now, this is a placeholder that returns None
    # A full Winograd implementation would be complex
    # Let's try a simpler structured approach
    return None


def _try_symmetric_decomposition(n, m, p):
    """
    Try to exploit the symmetry between the two dimension-5 indices.
    
    Since m = p = 5, we can try decompositions that treat j and k symmetrically.
    """
    # Placeholder - return None for now
    return None


def _try_rank_reduction(n, m, p, base_result, base_rank):
    """
    Try to reduce the rank of the base decomposition by finding linear dependencies
    among the rank-1 terms.
    """
    # Placeholder - return None for now
    return None
# EVOLVE-BLOCK-END
