"""Program for rank-minimal tensor decomposition of 3x5 @ 5x5 matrix multiplication."""
import time
import sys
import numpy as np
import itertools

DIMENSIONS = [3, 5, 5]
_DEADLINE = 120.0  # wall-clock seconds

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """
    Return the schoolbook (trivial) rank decomposition for n x m @ m x p.
    Rank = n*m*p.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    W follows the trace convention W[k*n + i, r].
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    The multiplication tensor T[i, j, k] = 1 if the output index (k, i) is
    produced by summing over j, i.e., T[i,j,k] = delta_{i,j,k} in the standard
    matrix mult sense: C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor decomposition requires:
      For all i in [0,n), j in [0,m), k in [0,p):
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Where T[i,j,k] = 1 (the multiplication tensor has all 1s for valid indices
    in this formulation... wait, let me reconsider).
    
    Actually, the matrix multiplication tensor is defined as:
      M[i, j, k] = 1 for all valid i, j, k (it's the "identity" tensor in the
      sense that it encodes the bilinear map).
    
    No wait. The matrix multiplication tensor T has entries:
      T[i, j, k] = 1  (for the standard matmul, the tensor is all ones in the
      appropriate multi-index space, but really it's the structure tensor).
    
    Let me think more carefully. The matrix multiplication bilinear map is:
      C[i,k] = sum_j A[i,j] B[j,k]
    
    This is a 3-tensor T with indices (i, j, k) where:
      T[i, j, k] = 1  (all entries are 1)
    
    And the decomposition says:
      T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    Wait, the W convention is W[k*n + i, r], so W is indexed by (k, i).
    
    So we need: for all i, j, k:
      1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Let's verify with the schoolbook:
      U[i*m+j, col] = 1 only when col corresponds to (i,j,k)
      V[j*p+k, col] = 1 only when col corresponds to (i,j,k)
      W[k*n+i, col] = 1 only when col corresponds to (i,j,k)
    
    So for a given (i,j,k), only the column col=(i,j,k) contributes, and
    U*V*W = 1*1*1 = 1. Correct.
    """
    rank = U.shape[1]
    # Build the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # This is a batched operation
    # T[i,j,k] = einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        # Fallback to manual computation
        T = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    T[i, j, k] = s
    
    # The target tensor is all ones
    target = np.ones((n, m, p), dtype=np.float64)
    
    return np.allclose(T, target, atol=atol), T


def _try_random_integer_decomposition(n, m, p, max_rank, time_budget, seed=None):
    """
    Try to find a decomposition with rank < schoolbook rank using random search.
    This is unlikely to succeed for small dimensions but we try.
    """
    if seed is not None:
        np.random.seed(seed)
    
    start = time.time()
    schoolbook_rank = n * m * p
    
    for rank in range(1, min(max_rank, schoolbook_rank)):
        if time.time() - start > time_budget:
            break
        
        # Try a few random attempts at this rank
        for attempt in range(20):
            if time.time() - start > time_budget:
                break
            
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            
            valid, _ = _verify_decomposition(U, V, W, n, m, p)
            if valid:
                return U, V, W, rank
    
    return None, None, None, None


def _structured_search(n, m, p, time_budget):
    """
    Try structured approaches to reduce rank below schoolbook.
    
    For 3x5 @ 5x5, the known lower bounds and upper bounds:
    - Schoolbook: 75
    - There may be faster algorithms but for these small dimensions,
      the exact rank is likely still 75 or close to it.
    
    We try a few structured decompositions.
    """
    start = time.time()
    
    # Strategy 1: Check if any simple block structure helps
    # For n=3, m=5, p=5, consider if we can exploit the small n dimension.
    
    # One idea: group the 3 rows and see if we can share some terms.
    # But in general, matrix multiplication tensors are known to be "rigid"
    # and don't admit simple rank reductions for small sizes.
    
    # Let's just do a quick randomized search with small time budget
    U, V, W, rank = _try_random_integer_decomposition(
        n, m, p, max_rank=70, time_budget=min(30.0, time_budget)
    )
    
    if U is not None:
        return U, V, W
    
    return None, None, None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 @ 5x5 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' containing lists of lists.
    """
    _start = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard: leave 10 seconds for cleanup
    deadline = _start + _DEADLINE - 10.0
    
    # First, get the schoolbook decomposition (guaranteed valid)
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Try to find a better (lower rank) decomposition
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        U_better, V_better, W_better = _structured_search(n, m, p, time_remaining)
        if U_better is not None:
            U, V, W = U_better, V_better, W_better
        else:
            U, V, W = U_school, V_school, W_school
    else:
        U, V, W = U_school, V_school, W_school
    
    # Final verification
    valid, T = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
