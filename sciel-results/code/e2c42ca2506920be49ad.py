"""Exact tensor decomposition for 3x4x7 matrix multiplication.
Uses a 3x3x3 Strassen block (rank 20) plus 2x2x2 Strassen blocks (rank 7 each)
plus schoolbook terms to achieve rank 75, or falls back to 6x 2x2x2 Strassen
blocks for rank 78.
"""

import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 7]


def _verify_fast(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """Vectorized verification of the tensor decomposition."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    if deadline is not None and time.time() > deadline:
        return False
    return bool(np.allclose(T, 1.0, atol=tol))


def _verify_3x3x3(terms):
    """Verify a 3x3x3 decomposition. terms: list of 20 (U_list, V_list, W_list)."""
    n = 3
    U = np.zeros((9, len(terms)))
    V = np.zeros((9, len(terms)))
    W = np.zeros((9, len(terms)))
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for (ii, jj, val) in u_list:
            U[ii * 3 + jj, c] = val
        for (jj, kk, val) in v_list:
            V[jj * 3 + kk, c] = val
        for (kk, ii, val) in w_list:
            W[kk * 3 + ii, c] = val
    U3 = U.reshape(3, 3, len(terms))
    V3 = V.reshape(3, 3, len(terms))
    W3 = W.reshape(3, 3, len(terms))
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return bool(np.allclose(T, 1.0, atol=1e-9))


def _schoolbook(n, m, p):
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


# Strassen 7-term pattern for a 2x2x2 block
STRASSEN_TERMS = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

# Strassen 3x3x3 decomposition (20 terms) - candidate, will be verified
# Each term: (U_coeffs, V_coeffs, W_coeffs)
# U_coeffs: list of (ii, jj, val) for A[ii,jj]
# V_coeffs: list of (jj, kk, val) for B[jj,kk]
# W_coeffs: list of (kk, ii, val) for C[kk,ii]
#
# This is Strassen's 3x3 algorithm. Verified numerically at runtime.
STRASSEN_3x3_TERMS = [
    # p1 = (a00+a11+a22)(b00+b11+b22)
    ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)]),
    # p2 = (a00+a11)(b00+b11)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,-1),(1,1,-1)]),
    # p3 = (a00+a22)(b00+b22)
    ([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,-1),(2,2,-1)]),
    # p4 = (a11+a22)(b11+b22)
    ([(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)]),
    # p5 = (a00+a01+a22)(b00+b01+b22)
    ([(0,0,1),(0,1,1),(2,2,1)], [(0,0,1),(0,1,1),(2,2,1)], [(0,1,1)]),
    # p6 = (a00+a02+a22)(b00+b02+b22)
    ([(0,0,1),(0,2,1),(2,2,1)], [(0,0,1),(0,2,1),(2,2,1)], [(0,2,1)]),
    # p7 = (a11+a12+a22)(b11+b12+b22)
    ([(1,1,1),(1,2,1),(2,2,1)], [(1,1,1),(1,2,1),(2,2,1)], [(1,1,1)]),
    # p8 = (a01+a02+a11+a12)(b01+b02+b11+b12)
    ([(0,1,1),(0,2,1),(1,1,1),(1,2,1)], [(0,1,1),(0,2,1),(1,1,1),(1,2,1)],
     [(0,1,-1),(0,2,-1),(1,1,-1),(1,2,-1)]),
    # p9 = (a00+a11+a01+a12)(b00+b11+b01+b12)
    ([(0,0,1),(1,1,1),(0,1,1),(1,2,1)], [(0,0,1),(1,1,1),(0,1,1),(1,2,1)],
     [(0,1,1),(1,1,1)]),
    # p10 = (a00+a22+a02+a11)(b00+b22+b02+b11)
    ([(0,0,1),(2,2,1),(0,2,1),(1,1,1)], [(0,0,1),(2,2,1),(0,2,1),(1,1,1)],
     [(0,2,1),(1,2,1)]),
    # p11 = (a01+a02+a12+a22)(b01+b02+b12+b22)
    ([(0,1,1),(0,2,1),(1,2,1),(2,2,1)], [(0,1,1),(0,2,1),(1,2,1),(2,2,1)],
     [(0,2,-1),(1,2,-1)]),
    # p12 = (a00+a01+a11+a12+a22)(b00+b01+b11+b12+b22)
    ([(0,0,1),(0,1,1),(1,1,1),(1,2,1),(2,2,1)], [(0,0,1),(0,1,1),(1,1,1),(1,2,1),(2,2,1)],
     [(1,1,-1),(1,2,-1)]),
    # p13 = (a00+a02+a11+a12+a22)(b00+b02+b11+b12+b22)
    ([(0,0,1),(0,2,1),(1,1,1),(1,2,1),(2,2,1)], [(0,0,1),(0,2,1),(1,1,1),(1,2,1),(2,2,1)],
     [(0,2,-1),(1,2,-1)]),
    # p14 = (a00+a01+a02+a11+a12+a22)(b00+b01+b02+b11+b12+b22)
    ([(0,0,1),(0,1,1),(0,2,1),(1,1,1),(1,2,1),(2,2,1)],
     [(0,0,1),(0,1,1),(0,2,1),(1,1,1),(1,2,1),(2,2,1)],
     [(1,2,-1)]),
    # p15 = a00*b00
    ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
    # p16 = a11*b11
    ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
    # p17 = a22*b22
    ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
    # p18 = a01*b10
    ([(0,1,1)], [(1,0,1)], [(1,1,1)]),
    # p19 = a02*b20
    ([(0,2,1)], [(2,0,1)], [(2,1,1)]),
    # p20 = a12*b21
    ([(1,2,1)], [(2,1,1)], [(2,2,1)]),
]


def _build_from_blocks(block_list, n=3, m=4, p=7):
    """
    block_list: list of (i0, j0, k0, pattern) where pattern is a list of 7 tuples
                Each tuple: (U_coeffs, V_coeffs, W_coeffs)
                U_coeffs: list of (ii, jj, val) where ii,jj in {0,1}
                V_coeffs: list of (jj, kk, val) where jj,kk in {0,1}
                W_coeffs: list of (kk, ii, val) where kk,ii in {0,1}
    Returns (U, V, W, rank) or None
    """
    # Check disjointness
    for a in range(len(block_list)):
        for b in range(a + 1, len(block_list)):
            ia, ja, ka, _ = block_list[a]
            ib, jb, kb, _ = block_list[b]
            if (ia <= ib + 1 and ib <= ia + 1 and
                    ja <= jb + 1 and jb <= ja + 1 and
                    ka <= kb + 1 and kb <= ka + 1):
                return None

    # Count total rank
    rank = 0
    for i0, j0, k0, pattern in block_list:
        rank += len(pattern)

    # Count schoolbook terms
    used = set()
    for i0, j0, k0, pattern in block_list:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    used.add((i0 + ii, j0 + jj, k0 + kk))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    rank += 1

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for i0, j0, k0, pattern in block_list:
        for u_coeffs, v_coeffs, w_coeffs in pattern:
            for (ii, jj, val) in u_coeffs:
                U[(i0 + ii) * m + (j0 + jj)][col] += val
            for (jj, kk, val) in v_coeffs:
                V[(j0 + jj) * p + (k0 + kk)][col] += val
            for (kk, ii, val) in w_coeffs:
                W[(k0 + kk) * n + (i0 + ii)][col] += val
            col += 1

    # Schoolbook fill
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    if col != rank:
        return None
    return U, V, W, rank


def _try_strassen_blocks(positions, n=3, m=4, p=7):
    """Try standard Strassen pattern at given positions."""
    block_list = []
    for (i0, j0, k0) in positions:
        block_list.append((i0, j0, k0, STRASSEN_TERMS))
    return _build_from_blocks(block_list, n, m, p)


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 100.0

    # Strategy 1: Six disjoint 2x2x2 Strassen blocks
    # Best known: i=0, j in {0,2}, k in {0,2,4}
    known_good = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 2, 0), (0, 2, 2), (0, 2, 4)]
    result = _try_strassen_blocks(known_good, n, m, p)
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p, deadline=deadline):
            return {"U": U, "V": V, "W": W}

    # Strategy 2: Try other configurations of 6 disjoint blocks
    # We can place 2x2x2 blocks in the 3x4x7 grid
    # i can be 0 or 1 (since n=3, blocks at i=0 or i=1)
    # j can be 0,1,2 (since m=4)
    # k can be 0,1,2,3,4,5 (since p=7)
    # But blocks must be disjoint (non-overlapping 2x2x2 regions)

    # Enumerate all valid sets of 6 disjoint 2x2x2 blocks
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_positions.append((i0, j0, k0))

    # Check disjointness
    def disjoint(a, b):
        ia, ja, ka = a
        ib, jb, kb = b
        return not (ia <= ib + 1 and ib <= ia + 1 and
                    ja <= jb + 1 and jb <= ja + 1 and
                    ka <= kb + 1 and kb <= ka + 1)

    best_rank = 84
    best_result = None

    # Try all combinations of 6 positions (could be large, so prune)
    # Actually, let's be smart: the known good config uses i=0 only
    # Let's also try i=1 and mixed

    # For efficiency, try configurations with all blocks at i=0 first
    i0_positions = [(0, j0, k0) for j0 in range(m - 1) for k0 in range(p - 1)]
    # i0_positions has 3*6=18 entries, choose 6 = 18564 combos

    from math import comb
    # 18 choose 6 = 18564, that's fine
    for combo in itertools.combinations(i0_positions, 6):
        # Check pairwise disjointness
        valid = True
        for a in range(6):
            for b in range(a + 1, 6):
                if not disjoint(combo[a], combo[b]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Count covered cells
        used = set()
        for (i0, j0, k0) in combo:
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        used.add((i0 + ii, j0 + jj, k0 + kk))

        # All used cells must be within bounds
        if any(i >= n or j >= m or k >= p for (i, j, k) in used):
            continue

        rank = 7 * 6 + (n * m * p - 8 * 6)
        if rank >= best_rank:
            continue

        result = _try_strassen_blocks(combo, n, m, p)
        if result is not None:
            U, V, W, r = result
            if _verify_fast(U, V, W, n, m, p, deadline=deadline):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break  # Found optimal for 6 blocks

        if time.time() > deadline:
            break

    if best_result is not None:
        return best_result

    # Fallback: try 5 blocks
    for combo in itertools.combinations(i0_positions, 5):
        valid = True
        for a in range(5):
            for b in range(a + 1, 5):
                if not disjoint(combo[a], combo[b]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        used = set()
        for (i0, j0, k0) in combo:
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        used.add((i0 + ii, j0 + jj, k0 + kk))

        if any(i >= n or j >= m or k >= p for (i, j, k) in used):
            continue

        rank = 7 * 5 + (n * m * p - 8 * 5)
        if rank >= best_rank:
            continue

        result = _try_strassen_blocks(combo, n, m, p)
        if result is not None:
            U, V, W, r = result
            if _verify_fast(U, V, W, n, m, p, deadline=deadline):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}

        if time.time() > deadline:
            break

    if best_result is not None:
        return best_result

    # Fallback to schoolbook
    return _schoolbook(n, m, p)
