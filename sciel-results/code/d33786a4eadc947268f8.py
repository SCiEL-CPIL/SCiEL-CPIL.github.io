"""Rewritten program: search for a low-rank exact tensor decomposition of 3x5 * 5x6."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS
N = n * m  # 15
M = m * p  # 30
P = p * n  # 18
SCHOOLBOOK_RANK = n * m * p  # 90

def _schoolbook():
    """Return the schoolbook decomposition (rank 90)."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N, rank), dtype=float)
    V = np.zeros((M, rank), dtype=float)
    W = np.zeros((P, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _to_lists(result):
    return {
        "U": [[float(x) for x in row] for row in result["U"]],
        "V": [[float(x) for x in row] for row in result["V"]],
        "W": [[float(x) for x in row] for row in result["W"]],
    }

def _check_exact(U, V, W, tol=1e-9):
    """
    Check if U, V, W exactly decompose the matrix multiplication tensor.
    U: (N, r), V: (M, r), W: (P, r)
    The tensor T[i,j,k] = delta(i,j,k) for matmul (3x5)*(5x6).
    We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 if valid, else 0.
    Actually the convention is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    Tensor T[i,j,k] = 1 always (it's the multiplication tensor).
    Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    """
    r = U.shape[1]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # Compute sum over c of Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    # This is a batched contraction
    # For each (i,j,k), compute sum_c Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    # We can do: for each c, add outer product contribution
    # More efficient: use einsum
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Note: Wr is indexed by (k, i, c)
    # Let's compute via einsum: 'ijc,jkc,kic->ijk'
    try:
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    except Exception:
        # Fallback: manual loop over c
        T = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            T += Ur[:, :, c] * Vr[:, :, c] * Wr[:, :, c].T  # Wr[k,i,c] -> shape (p,n), need (i,k)? 
            # Wait: Wr is (p, n, r) -> Wr[:, :, c] is (p, n)
            # We need Wr[k,i,c] for fixed i,k. 
            # T[i,j,k] += Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
            # Let's do it properly
            pass
        # Redo fallback correctly
        T = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            # Ur[:, :, c] is (n, m)
            # Vr[:, :, c] is (m, p)
            # Wr[:, :, c] is (p, n)
            # For each i,j,k: Ur[i,j]*Vr[j,k]*Wr[k,i]
            # This is: for each j, Ur[:,j]*Vr[j,:] gives (n,p), then multiply by Wr[k,i]
            # T[i,k] += sum_j Ur[i,j]*Vr[j,k]*Wr[k,i]... but Wr[k,i] depends on k,i
            # Let's just loop
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    
    # The multiplication tensor is all ones
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=tol, rtol=0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK
    
    # Try to find lower rank decompositions
    # For 3x5x6, known bounds: rank is between 3*5*6/something... 
    # The Strassen-like algorithms might help but for small matrices schoolbook is often near-optimal.
    # Let's try random initialization + optimization for ranks below 90.
    
    # First, let's try a few ranks: 89, 88, ... down to maybe 60
    # For each rank, try a few random initializations and optimize
    
    rng = np.random.default_rng(42)
    
    ranks_to_try = list(range(89, 59, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        for attempt in range(3):
            if time.time() > deadline:
                break
            # Random initialization
            U = rng.standard_normal((N, r))
            V = rng.standard_normal((M, r))
            W = rng.standard_normal((P, r))
            
            # Scale to avoid overflow
            U /= np.sqrt(np.sum(U**2))
            V /= np.sqrt(np.sum(V**2))
            W /= np.sqrt(np.sum(W**2))
            
            # Simple gradient-free: check if it's already close
            if _check_exact(U, V, W, tol=1e-8):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
        
        if best_rank < r:
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < SCHOOLBOOK_RANK:
        return _to_lists(best)
    
    return _to_lists(best)
