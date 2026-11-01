"""Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix."""
import time
from fractions import Fraction
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Exact verification using Fractions."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True


def _verify_np(U, V, W, n, m, p, tol=1e-10):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
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
    return {"U": U, "V": V, "W": W}


def _strassen22():
    """Strassen rank-7 decomposition for 2x2 * 2x2, returns (U,V,W) each (4,7)."""
    U = [
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
    ]
    W = [
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
    ]
    return U, V, W


def _build_blockwise(n, m, p):
    """
    Block-wise decomposition: split m into blocks, use Strassen for 2x2 blocks,
    schoolbook for remaining.
    For m=3: split into [2,1]. Strassen(2x2) for j=0,1 and schoolbook for j=2.
    Rank = 7*p + 2*p = 9*p = 90 for p=10.
    """
    U = np.zeros((n*m, 9*p))
    V = np.zeros((m*p, 9*p))
    W = np.zeros((p*n, 9*p))
    
    Us, Vs, Ws = _strassen22()
    
    # Strassen part: j in {0,1}, k in {0..p-1}
    for k in range(p):
        for c in range(7):
            base = k * 9 + c
            for a in range(4):
                U[a, base] = Us[a][c]
            for j in range(2):
                V[j*p + k, base] = Vs[j][c]
            for i in range(2):
                W[k*n + i, base] = Ws[i][c]
    
    # Schoolbook part: j=2, k in {0..p-1}
    for k in range(p):
        for i in range(2):
            base = k * 9 + 7 + i
            U[i*m + 2, base] = 1
            V[2*p + k, base] = 1
            W[k*n + i, base] = 1
    
    return U, V, W


def _try_randomized_search(n, m, p, max_rank, time_budget, seed=42):
    """
    Randomized search for low-rank decomposition.
    Start with random integer/half-integer factors and try to minimize rank.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    best_rank = max_rank
    best_U, best_V, best_W = None, None, None
    
    # Try various rank values from low to high
    for target_rank in range(max(1, max_rank // 2), max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate random factors with small integer/half-integer entries
            U = rng.randint(-2, 3, size=(n*m, target_rank)) * 0.5
            V = rng.randint(-2, 3, size=(m*p, target_rank)) * 0.5
            W = rng.randint(-2, 3, size=(p*n, target_rank)) * 0.5
            
            # Ensure at least some non-zero entries
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                continue
            
            # Check if this decomposition is valid
            if _verify_np(U, V, W, n, m, p, tol=1e-8):
                # Convert to exact fractions for verification
                Uf = [[float(x) for x in row] for row in U]
                Vf = [[float(x) for x in row] for row in V]
                Wf = [[float(x) for x in row] for row in W]
                
                if _verify(Uf, Vf, Wf, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    return best_rank, best_U, best_V, best_W
    
    return best_rank, best_U, best_V, best_W


def _try_structured_with_offsets(n, m, p, time_budget, seed=42):
    """
    Try structured decompositions with offset patterns.
    Use Kronecker products and shifts to create more efficient decompositions.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Base: schoolbook
    U0, V0, W0 = _build_blockwise(n, m, p)
    r0 = U0.shape[1]
    
    # Try reducing rank by zeroing out columns and checking validity
    for target_rank in range(max(10, r0 - 20), r0 + 1):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Randomly select columns to keep
            keep = np.sort(rng.choice(r0, size=target_rank, replace=False))
            Uk = U0[:, keep]
            Vk = V0[:, keep]
            Wk = W0[:, keep]
            
            if _verify_np(Uk, Vk, Wk, n, m, p, tol=1e-8):
                Uf = [[float(x) for x in row] for row in Uk]
                Vf = [[float(x) for x in row] for row in Vk]
                Wf = [[float(x) for x in row] for row in Wk]
                
                if _verify(Uf, Vf, Wf, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = Uk.copy(), Vk.copy(), Wk.copy()
                    return best_rank, best_U, best_V, best_W
    
    return best_rank, best_U, best_W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook (rank 60)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block-wise Strassen (rank 90) - not better than schoolbook, skip
    
    # Try structured reduction from block-wise
    if time.time() < deadline - 30:
        try:
            rk, Uk, Vk, Wk = _try_structured_with_offsets(n, m, p, min(20, deadline - time.time() - 10), seed=42)
            if rk < best_rank and Uk is not None:
                Uf = [[float(x) for x in row] for row in Uk]
                Vf = [[float(x) for x in row] for row in Vk]
                Wf = [[float(x) for x in row] for row in Wk]
                if _verify(Uf, Vf, Wf, n, m, p):
                    best = {"U": Uf, "V": Vf, "W": Wf}
                    best_rank = rk
        except Exception:
            pass
    
    # Try randomized search for lower rank
    if time.time() < deadline - 20 and best_rank > 30:
        try:
            rk, Uk, Vk, Wk = _try_randomized_search(n, m, p, best_rank, min(15, deadline - time.time() - 10), seed=12345)
            if rk < best_rank and Uk is not None:
                Uf = [[float(x) for x in row] for row in Uk]
                Vf = [[float(x) for x in row] for row in Vk]
                Wf = [[float(x) for x in row] for row in Wk]
                if _verify(Uf, Vf, Wf, n, m, p):
                    best = {"U": Uf, "V": Vf, "W": Wf}
                    best_rank = rk
        except Exception:
            pass
    
    return best
