"""Tensor decomposition search for 2x3 * 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the flattened matrix multiplication tensor T[i*n+j*m, k*p+l, i*n+k] is not quite right.
    Actually T has dimensions (n*m) x (m*p) x (p*n) following W[k*n+i, r].
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_decomposition(n, m, p, U, V, W):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each output C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    Tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    T = _build_tensor(n, m, p)
    r = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=np.float64) if isinstance(U, list) else U
    V_arr = np.array(V, dtype=np.float64) if isinstance(V, list) else V
    W_arr = np.array(W, dtype=np.float64) if isinstance(W, list) else W
    
    # Reconstruct tensor: for each r, outer product of U[:,r], V[:,r], W[:,r]
    # T_rec = sum_r U[:,r] V[:,r] W[:,r]
    recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    return np.allclose(recon, T, atol=1e-10)

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random rank decomposition attempt with integer/half-integer entries."""
    if seed is not None:
        rng = random.Random(seed)
        rng_np = np.random.RandomState(seed)
    else:
        rng = random.Random()
        rng_np = np.random.RandomState()
    
    def rand_entry():
        # Use small integers and half-integers
        vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        return rng.choice(vals)
    
    U = [[rand_entry() for _ in range(rank)] for _ in range(n*m)]
    V = [[rand_entry() for _ in range(rank)] for _ in range(m*p)]
    W = [[rand_entry() for _ in range(rank)] for _ in range(p*n)]
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
    start_time = time.time()
    deadline = 130.0  # Leave safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Build the tensor once
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook (guaranteed valid, rank 54)
    best_U, best_V, best_W, best_rank = _schoolbook(n, m, p)
    
    # Try to find lower rank by random search with structured patterns
    # The structure: A is 2x3, B is 3x9. Each output C[k*n+i] depends on A[i,:] and B[:,k]
    # C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    
    # Key insight: For each j (middle index), the contribution is A[i,j] * B[j,k] for all i,k.
    # This is a rank-2 tensor per j (outer product of column j of A and column k of B)...
    # Actually for each j, the contribution to the full tensor is:
    #   For all i,k: T[i,j,k] += A[i,j] * B[j,k]
    # The tensor slice for fixed j is the outer product of A[:,j] and B[j,:]
    # This has rank 2 (since A[:,j] is 2-dim vector, B[j,:] is 9-dim vector, outer product has rank 1!)
    # Wait: A[:,j] is a 2x1 vector, B[j,:] is a 1x9 vector. Outer product is 2x9, rank 1.
    # So each j contributes rank 1 to each "block" but the full tensor couples i and k through j.
    
    # Actually, the full tensor T[i*m+j, j*p+k, k*n+i] = 1 when (i,j) and (j,k) and (k,i) are consistent.
    # The schoolbook uses rank 54. The question is whether we can do better.
    
    # For n=2, m=3, p=9: the matrix multiplication tensor M_{2,3,9}
    # Known results: M_{2,3,9} rank is at most 54 (schoolbook)
    # Strassen-like reductions may apply but for these small dimensions, 54 may be optimal
    # or close to optimal.
    
    # Try structured random search for rank < 54
    for rank in range(schoolbook_rank - 1, 30, -1):
        if time.time() > deadline - 10:
            break
        attempts = 0
        max_attempts = 2000
        while time.time() < deadline - 5 and attempts < max_attempts:
            attempts += 1
            if time.time() > deadline - 2:
                break
            U, V, W = _random_decomposition(n, m, p, rank)
            if _verify_decomposition(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Return schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
    """Classic schoolbook decomposition."""
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
    return U, V, W
# EVOLVE-BLOCK-END
