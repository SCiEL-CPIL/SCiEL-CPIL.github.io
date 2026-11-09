"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition.

Uses a rank-44 Strassen-based construction with targeted rank-reduction search.
Includes a deadline guard to ensure completion within 150s.
"""

import numpy as np
import time
from itertools import combinations

_T0 = time.perf_counter()
_DEADLINE = 145.0          # safety margin before the 150 s kill

def _time_left():
    return _DEADLINE - (time.perf_counter() - _T0)

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _make_target(n, m, p):
    """Build the matrix-multiplication tensor T[u,v,w]=1 where
    u = i*m+j, v = j*p+k, w = k*n+i."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _check(U, V, W, n, m, p, tol=1e-12):
    """Verify sum_r U[:,r] V[:,r] W[:,r] == T  exactly (within tol)."""
    Tu = np.array(U, dtype=np.float64)
    Tv = np.array(V, dtype=np.float64)
    Tw = np.array(W, dtype=np.float64)
    T = _make_target(n, m, p)
    # contract over r:  result[u,v,w] = sum_r Tu[u,r]*Tv[v,r]*Tw[w,r]
    # do it efficiently:  (Tu @ Tv.T) gives (u,v); then for each w ...
    # Actually:  result = einsum('ur,vr,wr->uvw', Tu, Tv, Tw)
    R = np.einsum('ur,vr,wr->uvw', Tu, Tv, Tw, optimize=True)
    return bool(np.allclose(R, T, atol=tol, rtol=0.0))


# ---------------------------------------------------------------------------
# Rank-44 construction  (4 blocks of 2x3x2, each rank 11)
# ---------------------------------------------------------------------------

def _build_rank44():
    """Return (U, V, W) as lists-of-lists of ints for rank 44."""
    n, m, p = 2, 3, 8
    R = 44
    U = [[0] * R for _ in range(n * m)]   # 6 x 44
    V = [[0] * R for _ in range(m * p)]   # 24 x 44
    W = [[0] * R for _ in range(p * n)]   # 16 x 44

    # Pre-compute the 7 Strassen coefficients for a 2x2x2 block
    # plus 4 outer-product terms for the 2x1x2 part.
    # For each block b (k0=2b, k1=2b+1):
    #
    # Strassen (7 terms) operate on A-submatrix rows 0,1 cols 0,1
    #   and B-submatrix rows 0,1 cols k0,k1
    #
    # Outer product (4 terms) operate on A col 2 and B row 2.

    # Strassen term definitions for a generic 2x2x2:
    #  M1=(a+d)(e+h) -> c00, c11
    #  M2=(c+d)e     -> c10, c11
    #  M3=a(f-h)     -> c01, c11
    #  M4=d(g-e)     -> c00, c10
    #  M5=(a+b)h     -> c00, c01
    #  M6=(c-a)(e+f) -> c11
    #  M7=(b-d)(g+h) -> c00
    # where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    #       e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    #
    # In our indexing for block b:
    #   a = U[0], b = U[1], c = U[3], d = U[4]
    #   e = V[0*p+k0], f = V[0*p+k1], g = V[1*p+k0], h = V[1*p+k1]
    #   c00 = W[k0*n+0], c01 = W[k1*n+0], c10 = W[k0*n+1], c11 = W[k1*n+1]

    strassen_defs = [
        # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
        # M1 = (a+d)(e+h)
        ((0, 4), (1, 1), (None, None), (1, 1), (None, None), (1, 1)),  # placeholder
        # M2 = (c+d)e
        ((3, 4), (1, 1), (None,), (1,), (None, None), (1, -1)),
        # M3 = a(f-h)
        ((0,), (1,), (None, None), (1, -1), (None, None), (1, 1)),
        # M4 = d(g-e)
        ((4,), (1,), (None, None), (1, -1), (None, None), (1, 1)),
        # M5 = (a+b)h
        ((0, 1), (1, 1), (None,), (1,), (None, None), (-1, 1)),
        # M6 = (c-a)(e+f)
        ((3, 0), (1, -1), (None, None), (1, 1), (None,), (1,)),
        # M7 = (b-d)(g+h)
        ((1, 4), (1, -1), (None, None), (1, 1), (None,), (1,)),
    ]

    col = 0
    for b in range(4):
        k0, k1 = 2 * b, 2 * b + 1
        # V indices for e,f,g,h
        e_i = 0 * p + k0
        f_i = 0 * p + k1
        g_i = 1 * p + k0
        h_i = 1 * p + k1
        # W indices
        w00 = k0 * n + 0
        w01 = k1 * n + 0
        w10 = k0 * n + 1
        w11 = k1 * n + 1

        # Fill in the None placeholders
        sf = [
            ((0, 4), (1, 1), (e_i, h_i), (1, 1), (w00, w11), (1, 1)),
            ((3, 4), (1, 1), (e_i,),      (1,),  (w10, w11), (1, -1)),
            ((0,),     (1,),  (f_i, h_i), (1, -1),(w01, w11), (1, 1)),
            ((4,),     (1,),  (g_i, e_i), (1, -1),(w00, w10), (1, 1)),
            ((0, 1),   (1, 1), (h_i,),    (1,),  (w00, w01), (-1, 1)),
            ((3, 0),   (1, -1),(e_i, f_i),(1, 1), (w11,),    (1,)),
            ((1, 4),   (1, -1),(g_i, h_i),(1, 1), (w00,),    (1,)),
        ]

        for (ui, uc, vi, vc, wi, wc) in sf:
            for idx, c in zip(ui, uc):
                U[idx][col] += c
            for idx, c in zip(vi, vc):
                V[idx][col] += c
            for idx, c in zip(wi, wc):
                W[idx][col] += c
            col += 1

        # 4 outer-product terms for A-col-2 * B-row-2
        a2 = 0 * m + 2   # U index for A[0,2]
        d2 = 1 * m + 2   # U index for A[1,2]
        e2 = 2 * p + k0  # V index for B[2,k0]
        f2 = 2 * p + k1  # V index for B[2,k1]

        for (ui, vi, wi) in [(a2, e2, w00), (a2, f2, w01), (d2, e2, w10), (d2, f2, w11)]:
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Schoolbook fallback (rank 48)
# ---------------------------------------------------------------------------

def _build_schoolbook():
    n, m, p = 2, 3, 8
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------




def _try_rank_reduction(U, V, W, n, m, p, base_rank, max_rank_reduction=2):
    """Try to remove individual rank-1 components or replace pairs with singles.
    
    Returns (U, V, W, new_rank) if a lower rank is found, else (None, None, None, base_rank).
    """
    if _time_left() < 5.0:
        return None, None, None, base_rank
    
    R = base_rank
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Try removing each single component
    for r in range(R):
        if _time_left() < 3.0:
            break
        
        U_test = np.delete(U_arr, r, axis=1)
        V_test = np.delete(V_arr, r, axis=1)
        W_test = np.delete(W_arr, r, axis=1)
        
        if _check(U_test, V_test, W_test, n, m, p):
            return U_test.tolist(), V_test.tolist(), W_test.tolist(), R - 1
    
    # Try replacing pairs of components with a single component
    # This is more expensive, so only do it if time permits
    if _time_left() < 10.0:
        return None, None, None, base_rank
    
    # For each pair, check if we can find a single rank-1 component that covers both
    # This requires solving a system, which is complex. Instead, try a heuristic:
    # Check if the sum of two components can be approximated by one.
    
    # A simpler approach: try all single removals first (done above), then try 
    # removing two components at a time if the first pass didn't find anything
    for r1 in range(R):
        if _time_left() < 2.0:
            break
        for r2 in range(r1 + 1, R):
            if _time_left() < 2.0:
                break
            
            # Remove both r1 and r2
            mask = np.ones(R, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            U_test = U_arr[:, mask]
            V_test = V_arr[:, mask]
            W_test = W_arr[:, mask]
            
            if _check(U_test, V_test, W_test, n, m, p):
                return U_test.tolist(), V_test.tolist(), W_test.tolist(), R - 2
    
    return None, None, None, base_rank


def construct():
    n, m, p = 2, 3, 8

    # Fast path: rank-44 Strassen
    cand = _build_rank44()
    if _check(cand["U"], cand["V"], cand["W"], n, m, p):
        best = cand
        best_rank = 44
        
        # Try to reduce rank further
        U_red, V_red, W_red, new_rank = _try_rank_reduction(
            cand["U"], cand["V"], cand["W"], n, m, p, best_rank
        )
        if U_red is not None and new_rank < best_rank:
            best = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = new_rank
    else:
        best = None
        best_rank = 48

    # Fallback: schoolbook rank 48
    fb = _build_schoolbook()
    if _check(fb["U"], fb["V"], fb["W"], n, m, p):
        if best is None or 48 < best_rank:
            best = fb
            best_rank = 48

    if best is not None:
        return best

    return fb
