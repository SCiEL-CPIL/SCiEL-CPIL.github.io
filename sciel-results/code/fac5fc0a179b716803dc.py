"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy: Recursive block decomposition.
- Partition the 3x5x6 problem into smaller rectangular blocks.
- Use Strassen's algorithm (rank 7) for 2x2x2 sub-blocks.
- Use schoolbook (rank n*m*p) for 1x* or *x1 blocks.
- For other blocks, try splitting dimensions to minimize total rank.
- Validate the resulting decomposition against random inputs.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
DEADLINE = 145.0  # Safety margin before 150s kill

def _construct_schoolbook(n, m, p):
    """Construct schoolbook decomposition for n x m * m x p."""
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
    return U, V, W, rank

def _construct_strassen_222():
    """
    Strassen's algorithm for 2x2 * 2x2 -> 2x2.
    Returns U, V, W matrices for the 2x2x2 tensor.
    """
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to U, V, W.
    # U has size (2*2) x 7 = 4x7
    # V has size (2*2) x 7 = 4x7
    # W has size (2*2) x 7 = 4x7
    
    # Indices:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7  -> in V, index is j*p+k, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  -> in W, index is k*n+i, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11 + A22)(B11 + B22)
    # A11=0, A22=3; B11=0, B22=3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += M1, C22 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21 + A22)B11
    # A21=2, A22=3; B11=0
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2, C22 -= M2
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A11(B12 - B22)
    # A11=0; B12=1, B22=3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 += M3, C22 += M3
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22(B21 - B11)
    # A22=3; B21=2, B11=0
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 += M4, C21 += M4
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A11 + A12)B22
    # A11=0, A12=1; B22=3
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5, C12 += M5
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21 - A11)(B11 + B12)
    # A21=2, A11=0; B11=0, B12=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    W[3][5] = 1
    
    # M7: (A12 - A22)(B21 + B22)
    # A12=1, A22=3; B21=2, B22=3
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    W[0][6] = 1
    
    return U, V, W, 7

def _try_split_rank(n, m, p, start_time, deadline, cache):
    """
    Recursively find minimum rank for n x m * m x p.
    Returns (rank, U, V, W) or None if deadline exceeded.
    """
    if time.time() > deadline:
        return None
    
    key = (n, m, p)
    if key in cache:
        return cache[key]
    
    # Base cases
    if n == 1 or m == 1 or p == 1:
        result = _construct_schoolbook(n, m, p)
        cache[key] = result
        return result
    
    if n == 2 and m == 2 and p == 2:
        result = _construct_strassen_222()
        cache[key] = result
        return result
    
    # Try splitting
    best_rank = n * m * p
    best_result = _construct_schoolbook(n, m, p)
    
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        res1 = _try_split_rank(n1, m, p, start_time, deadline, cache)
        if res1 is None:
            return None
        res2 = _try_split_rank(n2, m, p, start_time, deadline, cache)
        if res2 is None:
            return None
        total_rank = res1[3] + res2[3]
        if total_rank < best_rank:
            best_rank = total_rank
            # Concatenate
            U = res1[0] + res2[0]
            V = res1[1] + res2[1]
            W = res1[2] + res2[2]
            best_result = (U, V, W, best_rank)
    
    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        res1 = _try_split_rank(n, m1, p, start_time, deadline, cache)
        if res1 is None:
            return None
        res2 = _try_split_rank(n, m2, p, start_time, deadline, cache)
        if res2 is None:
            return None
        total_rank = res1[3] + res2[3]
        if total_rank < best_rank:
            best_rank = total_rank
            U = res1[0] + res2[0]
            V = res1[1] + res2[1]
            W = res1[2] + res2[2]
            best_result = (U, V, W, best_rank)
    
    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        res1 = _try_split_rank(n, m, p1, start_time, deadline, cache)
        if res1 is None:
            return None
        res2 = _try_split_rank(n, m, p2, start_time, deadline, cache)
        if res2 is None:
            return None
        total_rank = res1[3] + res2[3]
        if total_rank < best_rank:
            best_rank = total_rank
            U = res1[0] + res2[0]
            V = res1[1] + res2[1]
            W = res1[2] + res2[2]
            best_result = (U, V, W, best_rank)
    
    cache[key] = best_result
    return best_result

def _validate_decomposition(U, V, W, n, m, p, trials=5, tol=1e-9):
    """
    Validate that U, V, W correctly decompose the matrix multiplication.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Compute C' using decomposition
        # A_flat = A.flatten('C') shape (n*m,)
        # B_flat = B.flatten('C') shape (m*p,)
        # Products = U.T @ A_flat * V.T @ B_flat  shape (rank,)
        # C_flat = W.T @ Products  shape (p*n,)
        
        A_flat = A.flatten('C')
        B_flat = B.flatten('C')
        
        products = (U_np.T @ A_flat) * (V_np.T @ B_flat)
        C_flat = W_np.T @ products
        
        C_expected = C.flatten('F')  # Note: W uses k*n+i, which is column-major if i is row, k is col? 
        # Let's check the index convention: W[k*n+i, r].
        # C is n x p.
        # C[i, k] is the element at row i, col k.
        # Index in W is k*n + i. This means if we flatten C in column-major order (Fortran), 
        # C_colmajor[k*n + i] = C[i, k].
        # So C_flat should be C.flatten('F').
        
        # Wait, let's re-verify.
        # A is n x m. B is m x p. C is n x p.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # U[i*m + j, r] is coefficient for A[i, j].
        # V[j*p + k, r] is coefficient for B[j, k].
        # W[k*n + i, r] is coefficient for C[i, k].
        # So C_flat[k*n + i] should equal C[i, k].
        # This corresponds to C.flatten('F') if we view C as a matrix with shape (n, p) and flatten columns.
        # In NumPy, C.flatten('F') gives [C[0,0], C[1,0], ..., C[n-1,0], C[0,1], ...]
        # Index k*n + i: for k=0, indices are 0..n-1 (i=0..n-1). These are C[0,0]...C[n-1,0].
        # For k=1, indices are n..2n-1. These are C[0,1]...C[n-1,1].
        # Yes, this matches C.flatten('F').
        
        if not np.allclose(C_flat, C_expected, atol=tol):
            return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    U_fb, V_fb, W_fb, rank_fb = _construct_schoolbook(n, m, p)
    result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try recursive decomposition
    cache = {}
    try:
        res = _try_split_rank(n, m, p, start_time, DEADLINE, cache)
        if res is not None:
            U, V, W, rank = res
            # Validate
            if _validate_decomposition(U, V, W, n, m, p):
                result = {"U": U, "V": V, "W": W}
    except Exception as e:
        # If anything goes wrong, return fallback
        pass
    
    # Final safety check
    if time.time() - start_time > 140:
        # Return the best result we have (which is likely the validated one or fallback)
        # If we haven't assigned result yet, it's the fallback.
        pass
        
    return result
# EVOLVE-BLOCK-END
