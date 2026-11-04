"""Tensor decomposition for 4x4x5 matrix multiplication - structured search."""
import time
import random
import itertools
from copy import deepcopy

DIMENSIONS = [4, 4, 5]
DEADLINE = 150  # seconds
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each (i, j, k) in the multiplication tensor, check:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if i==j==k pattern
    # Actually, the tensor T[i,j,k] for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition requires: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a "valid" triple
    # Wait, let me reconsider the tensor structure.
    
    # The matrix multiplication tensor M has indices (i, j, k) where:
    # (A*B)[i, k] = sum_j A[i, j] * B[j, k]
    # So M[i, j, k] = 1 for all valid (i, j, k) triples, and 0 otherwise.
    # But actually for the tensor decomposition, we need:
    # M[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Wait, I need to be more careful. The tensor for multiplying an n×m matrix 
    # by an m×p matrix to get an n×p matrix has entries:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # and (A*B)[i, k] = sum_j A[i, j] * B[j, k]
    
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k), and we decompose as:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Actually, I think the convention might be different. Let me re-read:
    # "U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i, r]" - this suggests W is indexed by (k, i) where k is the output column, i is the output row
    
    # The standard tensor decomposition for matrix multiplication:
    # (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # As a tensor: T[i, j, k] = 1 for all valid indices
    # Decomposition: T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # With the flattening: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    n, m, p = n, m, p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def try_strassen_style(n, m, p):
    """Try to construct a decomposition using Strassen-like patterns."""
    # For square matrices, Strassen reduces 8 to 7 multiplications.
    # For 4x4x4, we can use recursive Strassen.
    # For 4x4x5, we need to handle the rectangular case.
    
    # Let's try a direct approach: search for small-rank decompositions
    # with integer or half-integer coefficients.
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try ranks from a reasonable minimum up to schoolbook
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But in practice, we can do better than schoolbook (80)
    
    # For 4x4x5, let's try ranks from 20 to 40
    for rank in range(20, min(41, n * m * p + 1)):
        if time_remaining() < 30:
            break
            
        # Try random initializations with small integer coefficients
        for attempt in range(100):
            if time_remaining() < 30:
                break
                
            # Initialize with small integers: -1, 0, 1, 2
            U = [[random.choice([-1, 0, 0, 0, 1, 1, 2]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 0, 0, 1, 1, 2]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 0, 0, 1, 1, 2]) for _ in range(rank)] for _ in range(p * n)]
            
            if check_decomposition(U, V, W, n, m, p):
                best_rank = rank
                best_decomp = (deepcopy(U), deepcopy(V), deepcopy(W))
                return {"U": U, "V": V, "W": W}
    
    return None

def try_systematic_search(n, m, p):
    """Systematic search for low-rank decompositions."""
    best_rank = n * m * p
    best_decomp = None
    
    # Start from schoolbook and try to merge terms
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    # Try to find linear dependencies to reduce rank
    # This is complex, so let's try a different approach:
    # Use the fact that matrix multiplication tensors have structure
    
    # For now, return schoolbook if nothing better found
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a good decomposition
    result = try_strassen_style(n, m, p)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
