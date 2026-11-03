"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import nnls

def _verify(U, V, W, n, m, p):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build(terms, n, m, p):
    """Accumulate (u_idx, v_idx, w_idx, val) terms into U, V, W matrices."""
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u_idx, v_idx, w_idx, val) in enumerate(terms):
        U[u_idx][c] += val
        V[v_idx][c] += val
        W[w_idx][c] += val
    return U, V, W

def _strassen_block(terms, i0, j0, k0, m, p, n):
    """Emit rank-7 Strassen terms for the 2x2x2 sub-block at (i0..i0+1, j0..j0+1, k0..k0+1)."""
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1))
    # M2 = (a10+a11)(b00) -> C10 += M2, C11 += M2
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    # M3 = (a10)(b10-b00) -> C10 -= M3, C11 -= M3
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), -1))
    # M4 = (a00)(b11-b01) -> C00 += M4, C01 += M4
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), -1))
    # M5 = (a00+a01)(b11) -> C01 += M5, C11 += M5
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1))
    # M6 = (a10+a11)(b01-b11) -> C10 -= M6, C11 -= M6
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+1), 1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), -1))
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), -1))
    terms.append(((i0+1)*m + (j0+1), (j0+1)*p + (k0+0), (k0+0)*n + (i0+1), 1))
    # M7 = (a00-a10)(b00+b01) -> C00 -= M7, C01 -= M7
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+0), 1))
    terms.append(((i0+0)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+0), 1))
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+0), (k0+0)*n + (i0+1), -1))
    terms.append(((i0+1)*m + (j0+0), (j0+0)*p + (k0+1), (k0+1)*n + (i0+1), -1))

def _schoolbook_block(terms, i0, i1, j0, j1, k0, k1, m, p, n):
    """Emit schoolbook terms for the rectangular sub-block [i0..i1]x[j0..j1]x[k0..k1]."""
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                terms.append((i * m + j, j * p + k, k * n + i, 1))

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank by finding a lower-rank approximation via SVD/NNLS.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    if time.time() > deadline - 10:
        return None
    
    rank = len(U[0])
    if target_rank >= rank:
        return None
    
    # Build the tensor T[i,j,k] from current decomposition
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    
    # Try to find a rank-r decomposition
    # We'll use a greedy approach: try to find r vectors that span the tensor
    # For now, use SVD on matricizations
    
    # Matricize T along the first mode: T_(1) of shape (n, m*p)
    T1 = T.reshape(n, m*p)
    U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
    
    # Check if the tensor can be approximated by rank r
    # This is a heuristic; we'll try to project onto the top r singular vectors
    if S1.shape[0] < target_rank:
        return None
    
    # Use the top target_rank singular vectors
    U_new = U1[:, :target_rank]
    
    # Now we need to find V_new and W_new such that the decomposition holds
    # This is a hard problem, so we'll use a simple heuristic
    # For now, just return None and rely on the structured decomposition
    
    return None

def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 3, 5, 6

    # ---- Schoolbook fallback (rank 90) ----
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    # ---- Strassen-based construction (target rank 84) ----
    # Partition the 3x5x6 tensor:
    #   j-blocks: {0,1}, {2,3}, {4}
    #   i-blocks: {0,1}, {2}
    #   k-blocks: {0,1}, {2,3}, {4,5}
    # Strassen (rank 7) applies where i_count=2, j_count=2, k_count=2.
    # There are 2 j-blocks x 1 i-block x 3 k-blocks = 6 Strassen blocks -> 6*7 = 42 terms.
    # Remainder (i={2}) : 1 i x 5 j x 6 k = 30 schoolbook terms.
    # Total = 42 + 30 = 72? No, let's recompute carefully.
    
    # For j in {0,1}: i in {0,1}, k in {0,1},{2,3},{4,5} -> 3 Strassen blocks (2x2x2) = 21 terms
    # For j in {2,3}: i in {0,1}, k in {0,1},{2,3},{4,5} -> 3 Strassen blocks (2x2x2) = 21 terms
    # For j in {0,1}: i in {2}, k in {0,1},{2,3},{4,5} -> 3 schoolbook blocks (1x2x2) = 12 terms
    # For j in {2,3}: i in {2}, k in {0,1},{2,3},{4,5} -> 3 schoolbook blocks (1x2x2) = 12 terms
    # For j = 4: i in {0,1,2}, k in {0,1},{2,3},{4,5} -> 3 schoolbook blocks (3x1x2) = 18 terms
    # Total = 21 + 21 + 12 + 12 + 18 = 84 terms.

    terms = []
    for j0, j1 in [(0, 1), (2, 3)]:
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            _strassen_block(terms, 0, j0, k0, m, p, n)   # i0=0, i_count=2
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            _schoolbook_block(terms, 2, 2, j0, j1, k0, k1, m, p, n)  # i=2
    # j = 4 (single)
    for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
        _schoolbook_block(terms, 0, 2, 4, 4, k0, k1, m, p, n)

    U, V, W = _build(terms, n, m, p)

    # Verify exactness; fall back to schoolbook if anything is off.
    if _verify(U, V, W, n, m, p) and len(terms) < rank_fb:
        # Try to reduce rank further using numerical methods
        if time.time() < deadline - 20:
            for target_rank in range(len(terms) - 1, max(60, len(terms) - 10), -1):
                result = _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline)
                if result is not None:
                    U_new, V_new, W_new = result
                    if _verify(U_new, V_new, W_new, n, m, p):
                        return {"U": U_new, "V": V_new, "W": W_new}
        
        return {"U": U, "V": V, "W": W}

    return fallback
# EVOLVE-BLOCK-END
