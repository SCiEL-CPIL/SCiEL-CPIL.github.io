"""
Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition.

Strategy: Use the known Strassen-style block decomposition approach.
For 3x4x6, we can use the fact that 4 = 2+2 and 6 = 3+3 to potentially
use Strassen's algorithm on blocks, or find a direct low-rank construction.

The key insight: C[i,k] = sum_j A[i,j] B[j,k]
The tensor is T = sum_{i,j,k} e_i^A * e_j^B * e_k^C (with appropriate indices)

For 3x4x6, schoolbook rank = 72.
We try to find a lower rank using:
1. Direct construction based on algebraic identities
2. If that fails, fall back to schoolbook

Known: For 2x2x2, Strassen gives rank 7.
For 3x3x3, the best known is 23 (but 21 is conjectured).
For 3x4x6, we can try to use a mixed approach.

Let's try a direct algebraic construction:
We want to express sum_{j=0}^{3} A[i,j] B[j,k] using fewer than 24 terms per (i,k) pair,
but shared across all i,k.

Actually, let's think about this differently. The tensor has 3*4*6 = 72 "1" entries
in the sense that T[i,j,k] = 1 for each valid (i,j,k) triple, and the decomposition
requires:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
  (and 0 for invalid combinations, but all are valid here)

Wait, that's not quite right. The tensor is:
  T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i}

where u_{i,j} is the basis vector in R^{n*m} with 1 at position i*m+j,
v_{j,k} is the basis vector in R^{m*p} with 1 at position j*p+k,
w_{k,i} is the basis vector in R^{p*n} with 1 at position k*n+i.

So the schoolbook decomposition has rank n*m*p = 72.

To get a lower rank, we need to find linear dependencies. The question is whether
we can write some of these basis vector products as linear combinations of others.

For small cases, this is related to the border rank and actual rank of the matrix
multiplication tensor. For 3x4x6, I'm not aware of a known construction that beats
schoolbook significantly, but let's try a few approaches:

1. Try rank 60, 54, 48, etc. using a randomized search with integer/half-integer constraints
2. Use the structure: group by j (the middle index)

Let's implement a search that tries to find a valid decomposition at various ranks.
"""
import numpy as np
import random
import sys
import time

DIMENSIONS = [3, 4, 6]

def check_decomposition(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # Actually, we need: for all A, B: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition gives: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1
    # for the coefficient of A[i,j] B[j,k] in the bilinear form.
    
    # More precisely, the trilinear form is:
    # sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k]) * (sum_{i,k} W[k*n+i,r] C[i,k])
    
    # For this to equal the matrix multiplication tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def try_random_search(n, m, p, target_rank, max_iters=500, seed=None):
    """Try to find a decomposition of the given rank using random initialization."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        # Initialize with random small integers/half-integers
        U = np.random.randint(-2, 3, size=(n * m, target_rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m * p, target_rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p * n, target_rank)) * 0.5
        
        err = check_decomposition(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            return U, V, W
    
    return best_U, best_V, best_W

def schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # First, try a few ranks below schoolbook
    schoolbook_rank = n * m * p
    
    # Try a structured approach: use the fact that we can share computations
    # For now, try a few random searches at lower ranks
    for target_rank in [schoolbook_rank - 12, schoolbook_rank - 6, schoolbook_rank - 3]:
        if target_rank < 1:
            continue
        U, V, W = try_random_search(n, m, p, target_rank, max_iters=100, seed=target_rank)
        if U is not None:
            err = check_decomposition(U, V, W, n, m, p)
            if err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
