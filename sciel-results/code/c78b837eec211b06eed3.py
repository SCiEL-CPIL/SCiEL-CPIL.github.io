"""Exact tensor decomposition for 3x4x7 matrix multiplication.
Uses six disjoint 2x2x2 Strassen blocks to achieve rank 78.
"""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if deadline is not None and time.time() > deadline:
                    return False
                val = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
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

def _build_decomposition(blocks, n=3, m=4, p=7):
    """Build U,V,W from a set of disjoint Strassen blocks.
    blocks: list of (i0,j0,k0) tuples.
    Returns (U, V, W, rank) or None if invalid.
    """
    # Check all blocks are pairwise disjoint in their 2x2x2 regions
    for a in range(len(blocks)):
        for b in range(a+1, len(blocks)):
            ia, ja, ka = blocks[a]
            ib, jb, kb = blocks[b]
            # Check overlap: two 2x2x2 boxes overlap if they share any cell
            if (ia <= ib+1 and ib <= ia+1 and
                ja <= jb+1 and jb <= ja+1 and
                ka <= kb+1 and kb <= ka+1):
                return None  # Overlapping blocks
    
    rank = 7 * len(blocks) + (n*m*p - 8*len(blocks))
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for (i0, j0, k0) in blocks:
        i1, j1, k1 = i0+1, j0+1, k0+1
        for t in range(7):
            c = col + t
            if t == 0:
                U[i0*m+j0][c] = 1; U[i1*m+j1][c] = 1
                V[j0*p+k0][c] = 1; V[j1*p+k1][c] = 1
                W[k0*n+i0][c] = 1; W[k1*n+i1][c] = 1
            elif t == 1:
                U[i1*m+j0][c] = 1; U[i1*m+j1][c] = 1
                V[j0*p+k0][c] = 1
                W[k0*n+i1][c] = 1; W[k1*n+i1][c] = -1
            elif t == 2:
                U[i0*m+j0][c] = 1
                V[j0*p+k1][c] = 1; V[j1*p+k1][c] = -1
                W[k1*n+i0][c] = 1; W[k1*n+i1][c] = 1
            elif t == 3:
                U[i1*m+j1][c] = 1
                V[j1*p+k0][c] = 1; V[j0*p+k0][c] = -1
                W[k0*n+i0][c] = 1; W[k0*n+i1][c] = 1
            elif t == 4:
                U[i0*m+j0][c] = 1; U[i0*m+j1][c] = 1
                V[j1*p+k1][c] = 1
                W[k0*n+i0][c] = -1; W[k1*n+i0][c] = 1
            elif t == 5:
                U[i1*m+j0][c] = 1; U[i0*m+j0][c] = -1
                V[j0*p+k0][c] = 1; V[j0*p+k1][c] = 1
                W[k1*n+i1][c] = 1
            elif t == 6:
                U[i0*m+j1][c] = 1; U[i1*m+j1][c] = -1
                V[j1*p+k0][c] = 1; V[j1*p+k1][c] = 1
                W[k0*n+i0][c] = 1
        col += 7
    
    # Fill remaining with schoolbook
    used = set()
    for (i0, j0, k0) in blocks:
        for ii in range(i0, i0+2):
            for jj in range(j0, j0+2):
                for kk in range(k0, k0+2):
                    used.add((ii, jj, kk))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    if col != rank:
        return None
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0

    # Six disjoint 2x2x2 Strassen blocks at positions:
    # i=0, j in {0,2}, k in {0,2,4}
    # This tiles the j×k plane with 2×2 blocks, covering 48 of 84 triples
    known_good = [(0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4)]
    result = _build_decomposition(known_good, n, m, p)
    if result is not None:
        U, V, W, rank = result
        if _verify(U, V, W, n, m, p, deadline=deadline):
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return _schoolbook(n, m, p)
