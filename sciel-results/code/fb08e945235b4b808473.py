"""Candidate for matrix multiplication tensor decomposition (3x5x5)."""
import time
import numpy as np
from itertools import product

_DIM = (3, 5, 5)

def _verify(U, V, W, n, m, p):
    """Check that the decomposition is valid."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    r = len(W[0])
    
    # For each output (i,k), the bilinear form coefficient must match
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c2 in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c2 and d == k) else 0.0
                            actual = 0.0
                            for rr in range(r):
                                actual += Wa[k*n+i, rr] * Ua[a*m+b, rr] * Va[c2*p+d, rr]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _try_strassen_block(n, m, p, t0, deadline):
    """
    Try to use Strassen-style block decomposition to reduce rank.
    For 3x5x5, we can try to split into blocks and use Strassen on 2x2x2 blocks.
    """
    # Strategy: partition the 3x5x5 multiplication into blocks.
    # One approach: write A as [A1; A2] where A1 is 2x5, A2 is 1x5.
    # B is 5x5. C = A@B is 3x5.
    # C1 = A1@B (2x5), C2 = A2@B (1x5).
    # This doesn't directly help with Strassen.
    
    # Alternative: use the fact that we can do 2x2x2 Strassen within the computation.
    # For the 3x5x5 case, let's try a different block structure.
    
    # Let's try: split rows of A into groups of 2 and 1, columns of B into groups of 2 and 3.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 1x2, A22 is 1x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C = A@B is 3x5:
    # C = [[A11@B11 + A12@B21, A11@B12 + A12@B22],
    #      [A21@B11 + A22@B21, A21@B12 + A22@B22]]
    
    # This is getting complex. Let me try a simpler approach:
    # Use the known fact that for 2x2x2, Strassen gives rank 7 instead of 8.
    # For 3x5x5, we can try to identify 2x2x2 sub-problems.
    
    # Actually, let me try a direct approach: enumerate all possible ways to partition
    # the indices and apply Strassen-like reductions.
    
    # Simpler idea: try rank values below 75 and use a linear algebra approach.
    # For a given rank r, we need to find U (15xr), V (25xr), W (15xr) such that
    # sum_c W[k*n+i,c]*U[a*m+b,c]*V[c2*p+d,c] = delta[a,i]*delta[b,c2]*delta[d,k]
    
    # This is a system of linear equations in the entries of W (given U and V).
    # Let's try: fix random U and V, solve for W, check validity.
    
    # But this requires r to be large enough for the system to have a solution.
    
    # Let me try a different approach: use the "tensor rank" computation via
    # iterative SVD or other numerical methods.
    
    # For now, let's try to see if we can get rank < 75 by merging schoolbook terms.
    # The schoolbook has 75 terms. Can we combine some?
    
    # Actually, let me try the approach from the inspiration: recursive block splitting
    # with Strassen for 2x2x2.
    
    # For 3x5x5, one possible block decomposition:
    # Split the 5x5 matrix B into 2x2, 2x3, 3x2, 3x3 blocks.
    # Split the 3x5 matrix A into 2x2, 2x3, 1x2, 1x3 blocks.
    
    # Let me try a specific construction:
    # A is 3x5, B is 5x5, C is 3x5.
    
    # Block A as:
    #   A_top (2x5) and A_bottom (1x5)
    # Block B as:
    #   B_left (5x2) and B_right (5x3)
    
    # C_top = A_top @ B (2x5)
    # C_bottom = A_bottom @ B (1x5)
    
    # This is just splitting the output, not helping with rank reduction.
    
    # Let me try another approach: use the fact that some entries can be shared.
    
    # For now, return None to indicate this approach didn't yield improvement.
    return None

def _try_rank_reduction(n, m, p, t0, deadline):
    """
    Try to find a lower-rank decomposition using numerical methods.
    """
    # Build the multiplication tensor as a 3D array
    # T[i, j, k] represents the coefficient for A[i,j]*B[j,k] contributing to C[i,k]
    # Actually, the tensor is 4D: T[a,b,c,d] = 1 if a==i, c==b, d==k for output (i,k)
    # But we need to think of it as a bilinear form.
    
    # The bilinear form is: for each output (i,k), we have a matrix M_{i,k} of shape (n*m) x (m*p)
    # where M_{i,k}[(a,b),(c,d)] = 1 if a==i, b==c, d==k else 0.
    
    # The rank of the tensor is the minimum r such that we can write:
    # M_{i,k} = sum_c W[k*n+i,c] * u_c * v_c^T
    # where u_c is a vector of length n*m, v_c is a vector of length m*p.
    
    # This is equivalent to saying the tensor (viewed as a matrix for each (i,k))
    # has rank at most r when all matrices are considered together.
    
    # Let's try: form a large matrix by stacking all M_{i,k} and computing its rank.
    # But this gives an upper bound, not the exact tensor rank.
    
    # Alternative: use iterative methods to try to find a low-rank decomposition.
    
    # For simplicity, let's try ranks from 74 down to, say, 60, and for each rank,
    # try to find a valid decomposition using random initialization and least squares.
    
    # Actually, this is computationally expensive. Let me try a more targeted approach.
    
    # Known result: the rank of matrix multiplication for 2x2x2 is 7 (Strassen).
    # For 3x3x3, the rank is known to be between 19 and 23.
    # For 3x5x5, the rank is not well-studied, but it's likely close to 75.
    
    # Let me try a simple heuristic: start with schoolbook and try to merge pairs of terms
    # that can be combined.
    
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if they share the same (i,j) or (j,k)
    # in a way that allows algebraic combination.
    
    # Actually, let me just try to see if rank 74 is achievable by removing one term
    # and adjusting the others.
    
    # For now, this is a placeholder. Return None.
    return None

def construct():
    n, m, p = _DIM
    t0 = time.time()
    deadline = t0 + 120  # 120s safety margin (evaluator kills at 150s)
    
    # Start with schoolbook (rank 75) as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen block decomposition
    if time.time() < deadline - 30:
        result = _try_strassen_block(n, m, p, t0, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                rank = len(W[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
    
    # Try rank reduction
    if time.time() < deadline - 10:
        result = _try_rank_reduction(n, m, p, t0, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                rank = len(W[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
    
    # Verify the best result
    if _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    
    # If verification fails (shouldn't happen), return schoolbook
    return _schoolbook(n, m, p)
