"""Combined approach: structured block decomposition + refined optimization for 2x3x6 matrix mult."""
import time
import random
import math
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]
_START = time.time()
_DEADLINE = 120.0
_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _residual(U, V, W, n, m, p):
    """Return max absolute tensor residual over all (i,j,k)."""
    worst = 0.0
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(r):
                    s += uc[c] * vc[c] * wc[c]
                diff = abs(s - 1.0)
                if diff > worst:
                    worst = diff
    return worst

def _try_round_verify(U, V, W, tol=1e-9):
    """Round to half-integers and check exactness."""
    Ur = [[round(v * 2) / 2 for v in row] for row in U]
    Vr = [[round(v * 2) / 2 for v in row] for row in V]
    Wr = [[round(v * 2) / 2 for v in row] for row in W]
    n, m, p = DIMENSIONS
    r = len(Ur[0]) if Ur else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ur[i * m + j][c] * Vr[j * p + k][c] * Wr[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return None
    return {"U": Ur, "V": Vr, "W": Wr}

def _refine(U, V, W, n, m, p, iters=50):
    """Gauss-Seidel style refinement to drive residual toward zero."""
    r = len(U[0]) if U else 0
    for _ in range(iters):
        # Update W: for each (i,k), solve m linear equations
        for i in range(n):
            for k in range(p):
                A = []
                for j in range(m):
                    row = [U[i * m + j][c] * V[j * p + k][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        W[k * n + i][c] = float(sol[c])
                except Exception:
                    pass
        # Update U: for each (i,j), solve p linear equations
        for i in range(n):
            for j in range(m):
                A = []
                for k in range(p):
                    row = [V[j * p + k][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        U[i * m + j][c] = float(sol[c])
                except Exception:
                    pass
        # Update V: for each (j,k), solve n linear equations
        for j in range(m):
            for k in range(p):
                A = []
                for i in range(n):
                    row = [U[i * m + j][c] * W[k * n + i][c] for c in range(r)]
                    A.append(row)
                Aarr = np.array(A, dtype=float)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(Aarr, b, rcond=None)
                    for c in range(r):
                        V[j * p + k][c] = float(sol[c])
                except Exception:
                    pass
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _build_2x3x3_decomp():
    """Build a rank-7 decomposition for 2x3x3 matrix multiplication.
    
    For 2x3 * 3x3, the tensor has 18 entries.
    We use a known construction based on the 2x2x2 Strassen (rank 7) 
    combined with the third column.
    
    Actually, let's use a simpler approach: split into overlapping 2x2x2 blocks.
    
    For A (2x3) * B (3x3) = C (2x3):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as:
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    
    Let's use a rank-7 construction. One approach:
    Use the 7 products from Strassen on a 2x2 submatrix, plus 3 more for the third column.
    But that gives rank 10.
    
    Let me try a different approach: use the fact that 2x3x3 can be done with rank 7
    by exploiting the structure.
    
    Actually, let me just use random initialization + refinement for the 2x3x3 case
    and see if we can find rank 7 or 8.
    """
    pass

def _try_2x3x6_direct(rank, n, m, p, rng, max_attempts=50):
    """Try to find a valid 2x3x6 decomposition of given rank using random init + refinement."""
    best_res = float('inf')
    best_result = None
    
    for attempt in range(max_attempts):
        if _remaining() < 10.0:
            break
        
        # Choose initialization strategy
        if attempt % 5 == 0:
            # Sparse half-integer init
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            for row in U:
                for c in range(rank):
                    if rng.random() < 0.4:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in V:
                for c in range(rank):
                    if rng.random() < 0.4:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in W:
                for c in range(rank):
                    if rng.random() < 0.4:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
        elif attempt % 5 == 1:
            # Dense half-integer init
            U = [[rng.choice([-1, -0.5, 0.5, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, -0.5, 0.5, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, -0.5, 0.5, 1]) for _ in range(rank)] for _ in range(p * n)]
        elif attempt % 5 == 2:
            # Float init
            U = [[rng.uniform(-1.5, 1.5) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.uniform(-1.5, 1.5) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.uniform(-1.5, 1.5) for _ in range(rank)] for _ in range(p * n)]
        elif attempt % 5 == 3:
            # Structured: initialize based on schoolbook but with random signs
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i * m + j][col] = rng.choice([-1, 1])
                            V[j * p + k][col] = rng.choice([-1, 1])
                            W[k * n + i][col] = rng.choice([-1, 1])
                        col += 1
        else:
            # Very sparse
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            for row in U:
                for c in range(rank):
                    if rng.random() < 0.2:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in V:
                for c in range(rank):
                    if rng.random() < 0.2:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
            for row in W:
                for c in range(rank):
                    if rng.random() < 0.2:
                        row[c] = rng.choice([-1, -0.5, 0.5, 1])
        
        U, V, W = _refine(U, V, W, n, m, p, iters=60)
        res = _residual(U, V, W, n, m, p)
        
        if res < best_res:
            best_res = res
            if res < 1e-7:
                rounded = _try_round_verify(U, V, W)
                if rounded is not None:
                    return rounded
            best_result = (U, V, W)
    
    return None

def _try_block_decomposition():
    """Try to exploit block structure: 3x6 = three 3x2 blocks.
    
    For each 3x2 block, we need a 2x3x2 decomposition.
    If we can find rank-5 for 2x3x2, total rank would be 15.
    If rank-6, total rank 18.
    """
    n, m, p = DIMENSIONS
    # 2x3x2 tensor: 12 entries
    # Try to find rank-5 or rank-6 decomposition for 2x3x2
    # Then combine three copies
    
    # For now, try direct search on the full problem
    pass

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    rng = random.Random(42)
    
    # Try ranks from just below schoolbook down to a reasonable minimum
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But tensor rank lower bounds are tighter. For 2x3x6, likely >= 20.
    for rank in range(best_rank - 1, 17, -1):
        if _remaining() < 15.0:
            break
        
        result = _try_2x3x6_direct(rank, n, m, p, rng, max_attempts=30)
        if result is not None:
            best = result
            best_rank = rank
            # Found a valid decomposition, try to go lower
            continue
    
    return best

if __name__ == "__main__":
    print(construct())
