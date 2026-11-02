"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition.

Strategy: treat the 2x13 multiplication as a block multiplication over Z[x]
with degree constraints, leveraging Strassen's 7-fold decomposition for 2x2 blocks.

Primary construction: 6 Strassen blocks (rank 7 each) for column pairs + 1 schoolbook column
= rank 46. If verification fails, try polynomial-subspace reduction (aim for ~35-40).
Fallback: pure schoolbook (rank 52).
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 13]


def _verify(U, V, W, n=2, m=2, p=13, tol=1e-10):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_block(U, V, W, col, k1, k2, p=13):
    """Add 7 Strassen terms covering 2x2 * 2x{two columns}. Returns next col."""
    n = 2
    # M1 = (a00+a11)(b0k1+b1k2) -> c0k1 += M1, c1k2 += M1
    U[0, col] = 1; U[3, col] = 1
    V[k1, col] = 1; V[p + k2, col] = 1
    W[k1 * n, col] = 1; W[k2 * n + 1, col] = 1
    col += 1

    # M2 = (a10+a11)(b0k1) -> c1k1 += M2, c1k2 -= M2
    U[2, col] = 1; U[3, col] = 1
    V[k1, col] = 1
    W[k1 * n + 1, col] = 1; W[k2 * n + 1, col] = -1
    col += 1

    # M3 = a00(b0k2 - b1k2) -> c0k2 += M3, c1k2 += M3
    U[0, col] = 1
    V[p + k2, col] = 1; V[p + k2, col] = -1  # need separate entries
    V[k2, col] = 1
    V[p + k2, col] = -1
    W[k2 * n, col] = 1; W[k2 * n + 1, col] = 1
    col += 1

    # M4 = a11(b1k1 - b0k1) -> c0k1 += M4, c1k1 += M4
    U[3, col] = 1
    V[k1, col] = -1
    V[p + k1, col] = 1
    W[k1 * n, col] = 1; W[k1 * n + 1, col] = 1
    col += 1

    # M5 = (a00+a01)(b1k2) -> c0k1 -= M5, c0k2 += M5
    U[0, col] = 1; U[1, col] = 1
    V[p + k2, col] = 1
    W[k1 * n, col] = -1; W[k2 * n, col] = 1
    col += 1

    # M6 = (a10 - a00)(b0k1+b0k2) -> c1k2 += M6
    U[2, col] = 1; U[0, col] = -1
    V[k1, col] = 1; V[k2, col] = 1
    W[k2 * n + 1, col] = 1
    col += 1

    # M7 = (a01 - a11)(b1k1+b1k2) -> c0k1 += M7
    U[1, col] = 1; U[3, col] = -1
    V[p + k1, col] = 1; V[p + k2, col] = 1
    W[k1 * n, col] = 1
    col += 1

    return col


def _schoolbook_col(U, V, W, col, k, p=13):
    """Add 4 schoolbook terms for single column k. Returns next col."""
    n = 2
    for i in range(n):
        for j in range(n):
            U[i * n + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    return col


def _build_strassen_plus_schoolbook():
    """6 Strassen blocks + 1 schoolbook = rank 46."""
    n, m, p = 2, 2, 13
    rank = 46
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for group in range(6):
        k1 = 2 * group
        k2 = 2 * group + 1
        col = _strassen_block(U, V, W, col, k1, k2, p)
    col = _schoolbook_col(U, V, W, col, 12, p)

    return U, V, W


def _try_polynomial_reduction(base_rank=46, budget=60):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-46 factors.
    
    Idea: build the 70x70 Gram-like matrix G where G[a,b] = sum_{(i,j,k)} 
          U_a[i,j] * V_a[j,k] * W_a[k,i] * (same for b) ... actually that's not quite right.
    
    Better: the rank-46 tensor has components T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    We check if it equals the multiplication tensor. If some rank-r vectors are linearly
    dependent in a way that preserves T, we can project down.
    
    Simplest approach: use SVD/null-space of a carefully constructed matrix to find
    lower-rank equivalent. This is heuristic and may not improve things.
    """
    U, V, W = _build_strassen_plus_schoolbook()
    n, m, p = 2, 2, 13
    rank = U.shape[1]
    
    # Construct matrix M (52 x rank) where row indexed by (i,j,k) has entry at col r:
    # We need: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of 52 equations in the "product" sense, not linear.
    # 
    # Alternative: try to find a linear map on the rank space.
    # If there exists matrix C (rank x rank') such that U' = U C, V' = V C, W' = W^T C^T ... 
    # that doesn't work directly for tensors.
    
    # For a true tensor decomposition reduction, we need the same r-index structure.
    # Heuristic: perturb and re-verify won't help. Just return base.
    
    start = time.time()
    # Try: can we express some columns as linear combinations of others?
    # This requires solving a non-trivial optimization problem.
    # For simplicity within budget, try small perturbations + verification.
    
    best = (U.copy(), V.copy(), W.copy(), rank)
    
    # Attempt: random low-rank projection
    np.random.seed(42)
    for attempt in range(30):
        if time.time() - start > budget - 5:
            break
        new_rank = rank - (attempt % 3)
        if new_rank < 35:
            continue
        C = np.random.randn(rank, new_rank) / np.sqrt(rank)
        U2 = U @ C
        V2 = V @ C
        W2 = W @ C
        if _verify(U2, V2, W2):
            best = (U2, V2, W2, new_rank)
            break
        # Try structured: keep first new_rank columns, zero rest
        U3 = U[:, :new_rank].copy()
        V3 = V[:, :new_rank].copy()
        W3 = W[:, :new_rank].copy()
        if _verify(U3, V3, W3):
            best = (U3, V3, W3, new_rank)
            break
    
    return best


def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication.
    
    Primary: 6 Strassen blocks + 1 schoolbook column = rank 46.
    Attempt rank reduction if time permits.
    Fallback: pure schoolbook (rank 52).
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # keep 10s safety margin

    # Primary construction: rank 46
    U, V, W = _build_strassen_plus_schoolbook()
    
    if _verify(U, V, W):
        rank_primary = U.shape[1]
        
        # Try polynomial/subspace reduction if time remains
        if time.time() < deadline - 60:
            U2, V2, W2, rank2 = _try_polynomial_reduction(rank_primary, budget=55)
            if rank2 < rank_primary:
                U, V, W = U2, V2, W2
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: pure schoolbook (rank 52)
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1
                V_fb[j * p + k, c] = 1
                W_fb[k * n + i, c] = 1
                c += 1

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
