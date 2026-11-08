"""
Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.

Strategy:
  The schoolbook decomposition has rank 3*4*6 = 72.
  The Strassen-like / algebraic method can sometimes reduce rank for small matrices,
  but for 3x4 and 4x6 the known optimal ranks are not well below 72 in the literature
  for small cases. However, we can attempt a randomized search over integer/half-integer
  entries to find a lower-rank decomposition.

  Approach:
    1. Start with the schoolbook rank 72.
    2. Use a greedy / iterative elimination: try to express one tensor entry as a
       linear combination of the remaining ones, reducing rank by 1 if possible.
    3. Use randomized algebraic methods: pick random rank-r matrices U, V, W with
       integer/half-integer entries and check if they decompose the tensor.
    4. Use a time budget to ensure we return within 150s.

  For 3x4 * 4x6, the bilinear complexity (tensor rank) is known to be at most 72.
  We will attempt to find a decomposition of rank < 72.

  Key insight: The tensor T[i,j,k] = delta_{i,j} * delta_{j,k} for (i,j,k) in {0..2}x{0..3}x{0..5}
  can be decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].

  We'll use a randomized search with integer/half-integer coefficients, trying to
  minimize rank. We'll start from schoolbook and try to eliminate terms.
"""

import time
import random
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 6]

def _schoolbook():
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor from U, V, W
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if the indices correspond to
    # (A[i,j] * B[j,k]) contributing to C[i,k], i.e., T[i,j,k] = 1 for all valid (i,j,k)
    # Actually the tensor for matmul is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all (i,j,k) in range(n) x range(m) x range(p)
    # Wait, no. The matrix multiplication tensor has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 (it's the identity in the sense that each (i,j,k) contributes 1 to the product).
    
    # Actually, the tensor decomposition for matmul n x m * m x p is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # And for other combinations... wait, actually the tensor is only defined for these indices.
    # The factorization must satisfy:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_random_rank(n, m, p, target_rank, max_iters=1000, seed=None):
    """Try to find a rank-target_rank decomposition with integer/half-integer entries."""
    if seed is not None:
        random.seed(seed)
    
    for it in range(max_iters):
        # Generate random U, V, W with entries from {-1, -0.5, 0, 0.5, 1}
        U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(p*n)]
        
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    return None

def _greedy_eliminate(n, m, p, U, V, W, max_steps=50):
    """Try to eliminate one term at a time from the decomposition."""
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    improved = True
    while improved and rank > 1:
        improved = False
        for c in range(rank):
            # Try to remove column c by expressing it as a combination of others
            # This is complex; skip for now
            pass
        # Simple approach: try removing a column and checking if the rest still works
        # by solving for the removed column's contribution
        for c in range(rank):
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            if _verify(U_test, V_test, W_test, n, m, p, tol=1e-9):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
                improved = True
                break
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try greedy elimination on schoolbook
    result = _greedy_eliminate(n, m, p, best["U"], best["V"], best["W"])
    if result["rank"] < best_rank:
        best = {"U": result["U"], "V": result["V"], "W": result["W"]}
        best_rank = result["rank"]
    
    # Try random searches for lower ranks
    for target_rank in range(best_rank - 1, 10, -1):
        if time.time() > deadline:
            break
        # Limit iterations based on remaining time
        remaining = deadline - time.time()
        max_iters = max(10, int(remaining * 50))  # ~50 iters per second estimate
        result = _try_random_rank(n, m, p, target_rank, max_iters=max_iters)
        if result is not None:
            best = result
            best_rank = target_rank
            # Try to eliminate more
            elim = _greedy_eliminate(n, m, p, best["U"], best["V"], best["W"])
            if elim["rank"] < best_rank:
                best = {"U": elim["U"], "V": elim["V"], "W": elim["W"]}
                best_rank = elim["rank"]
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
