import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of tensor decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_2x2x2():
    """
    Strassen's 7-multiplication decomposition for 2x2x2.
    Returns list of 7 triples of (U_entries, V_entries, W_entries)
    where each entry is a list of (index, coeff) pairs.
    W uses trace convention: W[k*n + i, r]
    """
    terms = [
        # P1 = A11 * (B11 - B22)
        ([(0,0,1)], [(0,0,1),(1,1,-1)], [(0,0,1),(1,0,1)]),
        # P2 = (A11+A12) * B22
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1)]),
        # P3 = (A21+A22) * B11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]),
        # P4 = A22 * (B21-B11)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,1,1)]),
        # P5 = (A11+A22) * (B11+B22)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P6 = (A12-A22) * (B21+B22)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
        # P7 = (A21-A11) * (B11+B12)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,1,1)]),
    ]
    return terms

def _compute_block_rank(ns, ms, ps):
    """Compute the rank for a single block."""
    if ns == 2 and ms == 2 and ps == 2:
        return 7
    else:
        return ns * ms * ps

def _get_all_splits(dim, max_part=2):
    """Generate all compositions of dim into parts of size 1 and 2."""
    splits = []
    def _gen(remaining, current):
        if remaining == 0:
            splits.append(tuple(current))
            return
        for part in range(1, min(max_part, remaining) + 1):
            current.append(part)
            _gen(remaining - part, current)
            current.pop()
    _gen(dim, [])
    return splits

def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks, strassen_terms):
    """Build the full decomposition from block structure."""
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    # First pass: compute total rank
    total_rank = 0
    block_info = []
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                ns, ms, ps = n_blocks[ni], m_blocks[mi], p_blocks[pi]
                if ns == 2 and ms == 2 and ps == 2:
                    r = 7
                    method = 'strassen'
                else:
                    r = ns * ms * ps
                    method = 'trivial'
                total_rank += r
                block_info.append((ni, mi, pi, method, r))
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for ni, mi, pi, method, r in block_info:
        n0, m0, p0 = n_offsets[ni], m_offsets[mi], p_offsets[pi]
        ns, ms, ps = n_blocks[ni], m_blocks[mi], p_blocks[pi]
        
        if method == 'strassen':
            for a_factors, b_factors, c_factors in strassen_terms:
                for (ar, ac, acoeff) in a_factors:
                    U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                col += 1
        else:  # trivial
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n0 + i) * m + (m0 + j), col] = 1
                        V[(m0 + j) * p + (p0 + k), col] = 1
                        W[(p0 + k) * n + (n0 + i), col] = 1
                        col += 1
    
    assert col == total_rank
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    strassen_terms = _strassen_2x2x2()
    
    # Search over all block splits (parts of size 1 and 2)
    n_splits = _get_all_splits(n, max_part=2)
    m_splits = _get_all_splits(m, max_part=2)
    p_splits = _get_all_splits(p, max_part=2)
    
    best_rank = n * m * p
    best_split = ([n], [m], [p])
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = 0
                for (a, b, c) in zip(ns, ms, ps):
                    rank += _compute_block_rank(a, b, c)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (list(ns), list(ms), list(ps))
    
    if time.time() > deadline:
        U, V, W, rank = _build_decomposition(n, m, p, [n], [m], [p], strassen_terms)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the best decomposition
    U, V, W, rank = _build_decomposition(n, m, p, best_split[0], best_split[1], best_split[2], strassen_terms)
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: try specific splits
    fallback_splits = [
        ([2,1], [2,2,1], [2,2,2,1]),
        ([1,2], [2,2,1], [2,2,2,1]),
        ([2,1], [2,1,2], [2,2,2,1]),
        ([2,1], [2,2,1], [2,2,1,2]),
        ([1,2], [2,1,2], [2,2,2,1]),
        ([2,1], [2,1,2], [2,2,1,2]),
        ([1,2], [2,2,1], [2,2,1,2]),
        ([2,1], [2,2,1], [2,1,2,2]),
        ([1,2], [2,1,2], [2,1,2,2]),
        ([2,1], [2,1,2], [2,1,2,2]),
        ([1,2], [2,2,1], [2,1,2,2]),
        ([3], [5], [7]),
    ]
    
    for ns, ms, ps in fallback_splits:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _build_decomposition(n, m, p, ns, ms, ps, strassen_terms)
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue
    
    # Final fallback: schoolbook
    U, V, W, rank = _build_decomposition(n, m, p, [n], [m], [p], strassen_terms)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
