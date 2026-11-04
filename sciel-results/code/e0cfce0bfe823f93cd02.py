import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time as _time

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W for n=2, m=2, p=2 with rank 7."""
    # Standard Strassen:
    # M1 = (A11-A22)*(B22-B11)
    # M2 = (A11+A22)*B22
    # M3 = A11*(B11-B22)
    # M4 = A22*(B11+B22)
    # M5 = (A12+A22)*A21
    # M6 = (B11+B21)*B22
    # M7 = (A11-A12)*B12
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]  # 4 rows
    V = [[0]*rank for _ in range(m*p)]  # 4 rows
    W = [[0]*rank for _ in range(p*n)]  # 4 rows
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A11-A22)*(B22-B11) -> C11 += M1, C22 += M1
    U[0][0] = 1; U[3][0] = -1
    V[3][0] = 1; V[0][0] = -1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A11+A22)*B22 -> C21 += M2, C22 -= M2
    U[0][1] = 1; U[3][1] = 1
    V[3][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11*(B11-B22) -> C12 += M3, C22 += M3
    U[0][2] = 1
    V[0][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22*(B11+B22) -> C11 += M4, C21 += M4
    U[3][3] = 1
    V[0][3] = 1; V[3][3] = 1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A12+A22)*A21 -> C11 -= M5, C12 += M5
    U[1][4] = 1; U[3][4] = 1
    V[2][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (B11+B21)*B22 -> C22 += M6
    V[0][5] = 1; V[2][5] = 1
    V[3][5] = 1
    W[3][5] = 1
    
    # M7 = (A11-A12)*B12 -> C11 += M7
    U[0][6] = 1; U[1][6] = -1
    V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W, rank

def _block_strassen(n, m, p):
    """Use Strassen to reduce rank for block matrix multiplication.
    For 4x4 * 4x7, we can view as blocks.
    This is a heuristic attempt."""
    # For now, just return schoolbook
    return _schoolbook(n, m, p)

def _random_search(n, m, p, max_rank, max_iter, deadline):
    """Random search for low-rank decompositions with integer/half-integer entries."""
    best = None
    best_rank = max_rank
    
    # Start with schoolbook
    U0, V0, W0, r0 = _schoolbook(n, m, p)
    if _verify(U0, V0, W0, n, m, p):
        best = (U0, V0, W0)
        best_rank = r0
    
    rng = random.Random(42)
    
    # Try to find decompositions with rank < best_rank
    for rank in range(best_rank - 1, 63, -1):  # 63 = 3*7, a reasonable lower bound heuristic
        if _time.time() > deadline:
            break
        
        for trial in range(max_iter):
            if _time.time() > deadline:
                break
            
            # Generate random matrices with small integer/half-integer entries
            vals = [0, 1, -1, 0.5, -0.5]
            U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                best = (U, V, W)
                best_rank = rank
                break
    
    if best is not None:
        return best
    return U0, V0, W0

def _gaussian_elimination_search(n, m, p, deadline):
    """Try to find a lower-rank decomposition using linear algebra."""
    # The matrix multiplication tensor T has components T[i,j,k] = delta[i,j,k]
    # We want to write T as sum of rank-1 tensors
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # This is equivalent to finding a factorization of the 3-tensor
    # Let's try a different approach: use the fact that for fixed j,
    # the matrix A -> C is linear in A, so C = A * B
    # This means the tensor has a special structure.
    
    # For now, return schoolbook
    U, V, W, r = _schoolbook(n, m, p)
    return U, V, W, r

def construct():
    n, m, p = DIMENSIONS
    start = _time.time()
    deadline = start + 140  # Safety margin
    
    # Schoolbook baseline
    U, V, W, rank = _schoolbook(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        best = (U, V, W)
        best_rank = rank
    else:
        best = None
        best_rank = rank
    
    # Try Strassen-based block decomposition
    if _time.time() < deadline:
        try:
            Us, Vs, Ws, rs = _block_strassen(n, m, p)
            if _verify(Us, Vs, Ws, n, m, p) and rs < best_rank:
                best = (Us, Vs, Ws)
                best_rank = rs
        except:
            pass
    
    # Random search
    if _time.time() < deadline:
        try:
            Ur, Vr, Wr = _random_search(n, m, p, best_rank, 100, deadline)
            if Ur is not None:
                if _verify(Ur, Vr, Wr, n, m, p):
                    r = len(Ur[0])
                    if r < best_rank:
                        best = (Ur, Vr, Wr)
                        best_rank = r
        except:
            pass
    
    if best is not None:
        U, V, W = best
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
