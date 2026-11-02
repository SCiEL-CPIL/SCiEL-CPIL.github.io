"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIMEOUT = 140

# Strassen 2x2x2 decomposition (rank 7)
# Each entry is (U_indices, V_indices, W_indices) where indices are (i,j) for U (2x2), (j,k) for V (2x2), (k,i) for W (2x2)
STRASSEN_222 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

def build_block_decomp(n, m, p, j_groups):
    """
    Build U, V, W for a block decomposition of matrix multiplication.
    j_groups: list of tuples of j-indices. Each group is either size 1 (schoolbook) or size 2 (Strassen).
    Returns (U, V, W, rank)
    """
    rank = 0
    for grp in j_groups:
        sz = len(grp)
        if sz == 1:
            rank += n * p
        elif sz == 2:
            rank += (p // 2) * 7  # 3 * 7 = 21 for Strassen 2x2x2 with p=6 -> p//2=3
        else:
            raise ValueError("group size not supported")
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for grp in j_groups:
        sz = len(grp)
        if sz == 1:
            j0 = grp[0]
            for i in range(n):
                for k in range(p):
                    U[i * m + j0, col] = 1.0
                    V[j0 * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        elif sz == 2:
            j0, j1 = grp
            for k_pair in range(p // 2):
                k0 = k_pair * 2
                k1 = k_pair + 1
                for (u_list, v_list, w_list) in STRASSEN_222:
                    for (li, lj, val) in u_list:
                        U[li * m + (j0 + lj), col] += val
                    for (lj, lk, val) in v_list:
                        V[(j0 + lj) * p + (k0 + lk), col] += val
                    for (li, lk, val) in w_list:
                        W[(k0 + lk) * n + li, col] += val
                    col += 1
    return U, V, W, col

def verify_decomp(n, m, p, U, V, W):
    """Check if the decomposition is exact."""
    Ua = U.reshape(n, m, -1)
    Va = V.reshape(m, p, -1)
    Wa = W.reshape(p, n, -1)
    S = np.einsum('ijt,jkt,kit->ijk', Ua, Va, Wa)
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def generate_j_splits(m):
    """Generate all partitions of {0,...,m-1} into groups of size 1 or 2."""
    indices = list(range(m))
    
    def gen(rest):
        if not rest:
            yield []
            return
        for i in range(len(rest)):
            # Single
            r = rest[:i] + rest[i+1:]
            for sub in gen(r):
                yield [[rest[i]]] + sub
            # Pair
            if i + 1 < len(rest):
                r = rest[:i] + rest[i+2:]
                for sub in gen(r):
                    yield [[rest[i], rest[i+1]]] + sub
    
    seen = set()
    results = []
    for sp in gen(indices):
        key = tuple(sorted(tuple(sorted(g)) for g in sp))
        if key not in seen:
            seen.add(key)
            results.append(sp)
    return results

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    splits = generate_j_splits(m)
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    for sp in splits:
        if time.time() - _START_TIME > _TIMEOUT - 10:
            break
        U, V, W, col = build_block_decomp(n, m, p, sp)
        if verify_decomp(n, m, p, U, V, W):
            if best_rank is None or col < best_rank:
                best_rank = col
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
    
    if best_rank is None:
        # Fallback: naive
        best_rank = n * m * p
        best_U = np.zeros((n * m, best_rank), dtype=float)
        best_V = np.zeros((m * p, best_rank), dtype=float)
        best_W = np.zeros((p * n, best_rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1.0
                    best_V[j * p + k, col] = 1.0
                    best_W[k * n + i, col] = 1.0
                    col += 1
    
    def to_clean(arr):
        out = []
        for row in arr:
            out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
        return out
    
    return {"U": to_clean(best_U), "V": to_clean(best_V), "W": to_clean(best_W)}
# EVOLVE-BLOCK-END
