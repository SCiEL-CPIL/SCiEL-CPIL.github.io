"""Rewritten: search for rank-minimal exact tensor decomposition of 3x3 * 3x4 matmul."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    n, m, p = 3, 3, 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W):
    """Verify that U,V,W exactly decompose the matmul tensor."""
    n, m, p = 3, 3, 4
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    # Compute decomposition
    D = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(D, T, atol=1e-10)

def _verify_exact(U, V, W):
    """Exact verification using Fraction."""
    n, m, p = 3, 3, 4
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    # Compute decomposition exactly
    D = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                s = Fraction(0)
                for r in range(U.shape[1]):
                    s += Fraction(U[a, r]) * Fraction(V[b, r]) * Fraction(W[c, r])
                D[a, b, c] = s
    # Check
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if D[a, b, c] != T[a, b, c]:
                    return False
    return True

def _try_rank_r(rank, seed=42, time_limit=10.0):
    """Try to find a rank-r decomposition using random local search."""
    n, m, p = 3, 3, 4
    deadline = time.time() + time_limit
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Local search: try to minimize ||D - T||_F^2
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(5000):
        if time.time() > deadline:
            break
        D = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((D - T)**2)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-10:
            # Check exact
            if _verify_exact(U, V, W):
                return {"U": U, "V": V, "W": W}
            break
        
        # Perturb one random element
        mat = np.random.randint(3)
        idx = np.random.randint(U.shape[0] if mat == 0 else V.shape[0] if mat == 1 else W.shape[0])
        r = np.random.randint(rank)
        if mat == 0:
            U[idx, r] += np.random.randn() * 0.1
        elif mat == 1:
            V[idx, r] += np.random.randn() * 0.1
        else:
            W[idx, r] += np.random.randn() * 0.1
    
    if best_err < 1e-8:
        # Try to snap to integers/half-integers
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        if _verify_exact(U_snap, V_snap, W_snap):
            return {"U": U_snap, "V": V_snap, "W": W_snap}
    
    return None

def _strassen_block_2x2x2():
    """Return the 7 Strassen terms for 2x2x2 matmul tensor."""
    terms = [
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,1],[1,1]])),
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])),
        (np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]]), np.array([[0,0],[1,0]])),
        (np.array([[0,0],[0,1]]), np.array([[0,1],[-1,0]]), np.array([[0,1],[0,0]])),
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[0,1],[0,0]])),
        (np.array([[-1,0],[1,0]]), np.array([[1,0],[1,0]]), np.array([[0,0],[1,0]])),
        (np.array([[0,1],[-1,0]]), np.array([[0,1],[0,1]]), np.array([[0,1],[0,0]]))
    ]
    return terms

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    # Try rank 35: Strassen for 2x2x2 block + Schoolbook for rest
    rank = 35
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Embed Strassen terms for i,j,k in {0,1}
    terms = _strassen_block_2x2x2()
    col = 0
    for u, v, w in terms:
        for i in range(2):
            for j in range(2):
                U[i*m + j, col] = u[i, j]
        for j in range(2):
            for k in range(2):
                V[j*p + k, col] = v[j, k]
        for k in range(2):
            for i in range(2):
                W[k*n + i, col] = w[i, k]
        col += 1
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 34 with random local search
    if time.time() < deadline:
        result = _try_rank_r(34, seed=42, time_limit=20.0)
        if result is not None:
            return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    
    # Try rank 33
    if time.time() < deadline:
        result = _try_rank_r(33, seed=123, time_limit=20.0)
        if result is not None:
            return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    
    # Fallback to schoolbook rank 36
    fb = _schoolbook()
    return fb

# EVOLVE-BLOCK-END
