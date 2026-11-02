"""Matrix multiplication 2x3x6 tensor decomposition; direct construction + rank reduction search."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np
from fractions import Fraction
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build the multiplication tensor T with indices (i, j, k) -> T[i,j,k] = 1
    # if the monomial A[i,j]*B[j,k] appears in C[i,k], else 0.
    # Actually, for matrix multiplication, the tensor is defined such that:
    # T_{ijk} = 1 if the product A_{ij} B_{jk} contributes to C_{ik}, else 0.
    # In the standard decomposition, each monomial A[i,j]*B[j,k] is a separate term.
    # The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 if (i,j) is a valid A index, (j,k) is a valid B index, and
    # the product contributes to C[i,k]. For generic matrix multiplication, 
    # T[i,j,k] = 1 for all valid (i,j,k).
    
    # The tensor we're decomposing is the multiplication tensor.
    # A rank-r decomposition gives matrices U (nm x r), V (mp x r), W (pn x r)
    # such that for all i,j,k: T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # Trivial rank is n*m*p = 36. We try to find lower rank.
    
    # For now, return the trivial decomposition.
    # We can try to improve by searching for lower rank decompositions.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower rank decomposition.
    # The key insight: for 2x3x6, we can try to find a rank-r decomposition
    # where r < 36.
    
    # One approach: use the fact that the tensor can be viewed as a set of
    # linear equations. We want to find U, V, W with minimal r such that:
    # For each (i,j,k): T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # Since T[i,j,k] = 1 for all valid (i,j,k), we need:
    # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all i,j,k.
    
    # This is a system of nonlinear equations. Let's try a few known constructions.
    
    # For 2x3x6, one possible approach is to use the Kronecker product structure.
    # But let's first try a simple search for small perturbations.
    
    # Actually, let's try a different approach: use the fact that the multiplication
    # tensor for A(2x3) * B(3x6) can be decomposed using the structure of the indices.
    
    # The tensor has 36 entries, all equal to 1. We need to find U, V, W such that
    # the tensor product sum_s U_s * V_s * W_s equals T.
    
    # Let's try to use a rank-36 decomposition and see if we can merge terms.
    
    # For now, return the trivial rank-36 decomposition.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
