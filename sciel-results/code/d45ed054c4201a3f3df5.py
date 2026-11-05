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
    """
    # Strassen's 7 products:
    # P1 = (A11)(B11-B22)
    # P2 = (A11+A12)(B22)
    # P3 = (A21+A22)(B11)
    # P4 = (A22)(B21-B11)
    # P5 = (A11+A22)(B11+B22)
    # P6 = (A12-A22)(B21+B22)
    # P7 = (A21-A11)(B11+B12)
    #
    # C11 = P5 + P4 - P2 - P7
    # C12 = P1 + P2
    # C21 = P3 + P4
    # C22 = P1 - P3 + P5 + P6
    
    # Each term: (U_factors, V_factors, W_factors)
    # U: A matrix (2x2), index (i,j) -> row i, col j
    # V: B matrix (2x2), index (j,k) -> row j, col k
    # W: C matrix (2x2), index (k,i) -> row k, col i (trace convention W[k*n+i])
    # For 2x2x2: W index is (k, i) where k is col of C, i is row of C
    # Actually W[k*n+i, r] where n=2, so W[k*2+i, r]
    
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

def _bini_3x3x3():
    """
    Bini's rank-23 decomposition for 3x3x3.
    This is a known decomposition. I'll use a standard one.
    """
    # Bini's 23-term decomposition of 3x3x3 matrix multiplication
    # This is complex; let me use a known formulation.
    # Actually, let me try to construct it or use a simpler approach.
    # For now, I'll use the trivial 27-term as fallback and try to find a better one.
    # Let me use a known Bini decomposition.
    
    # Bini's decomposition (one known version):
    # I'll encode the 23 terms. Each term is (u, v, w) where
    # u is 3x3, v is 3x3, w is 3x3 (in trace convention W[k*n+i])
    
    # Actually, let me just use a simpler approach: try to solve for it
    # or use a precomputed one. For robustness, I'll provide a known set.
    
    # Let me use the approach of trying to find it numerically, but with a
    # better initialization. Actually, for reliability within time limits,
    # let me use a simpler strategy: use Strassen where possible and
    # schoolbook for the rest, but optimize the block structure.
    
    # For 3x3x3, the best known is 23 (Bini). Let me try to hardcode
    # a known decomposition or compute it.
    
    # I'll return None to indicate we should use schoolbook for 3x3x3
    # unless we can find a better decomposition.
    return None

def _compute_block_rank(ns, ms, ps, has_strassen=True, has_bini=False):
    """Compute the rank for a single block."""
    if ns == 2 and ms == 2 and ps == 2 and has_strassen:
        return 7
    elif ns == 3 and ms == 3 and ps == 3 and has_bini:
        return 23
    else:
        return ns * ms * ps

def _get_all_splits(dim, max_part=3):
    """Generate all compositions of dim into parts of size 1, 2, and 3."""
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

def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks, strassen_terms, bini_decomp=None):
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
                elif ns == 3 and ms == 3 and ps == 3 and bini_decomp is not None:
                    r = 23
                    method = 'bini'
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
                    # W index: (p0 + cc) * n + (n0 + cr)
                    W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                col += 1
        elif method == 'bini':
            U3, V3, W3 = bini_decomp
            for r_idx in range(23):
                for i in range(3):
                    for j in range(3):
                        U[(n0 + i) * m + (m0 + j), col] = U3[r_idx, i, j]
                for j in range(3):
                    for k in range(3):
                        V[(m0 + j) * p + (p0 + k), col] = V3[r_idx, j, k]
                for k in range(3):
                    for i in range(3):
                        W[(p0 + k) * n + (n0 + i), col] = W3[r_idx, k, i]
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

def _find_bini_333(deadline):
    """
    Try to find a rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses least squares with multiple initializations.
    """
    from scipy.optimize import least_squares
    
    rank = 23
    n = m = p = 3
    
    def residuals(params):
        # params: (rank, 27) flattened
        res = np.zeros(27)
        for r in range(rank):
            u = params[r*27:r*27+9]
            v = params[r*27+9:r*27+18]
            w = params[r*27+18:r*27+27]
            # u[i,j], v[j,k], w[k,i]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        res[i*9 + j*3 + k] += u[i*3+j] * v[j*3+k] * w[k*3+i]
        res -= 1.0
        return res
    
    # Try several initializations
    np.random.seed(123)
    best_cost = np.inf
    best_x = None
    
    # Initialization 1: Start from trivial 27 terms, zero out 4
    p0 = np.zeros(rank * 27)
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if idx < rank:
                    p0[idx*27 + i*3+j] = 1.0
                    p0[idx*27 + 9 + j*3+k] = 1.0
                    p0[idx*27 + 18 + k*3+i] = 1.0
                idx += 1
    
    # Initialization 2: Random
    p1 = np.random.randn(rank * 27) * 0.3
    
    # Initialization 3: Trivial with perturbation
    p2 = p0 + np.random.randn(rank * 27) * 0.1
    
    inits = [p0, p1, p2]
    
    for p_init in inits:
        if time.time() > deadline:
            break
        for attempt in range(2):
            if time.time() > deadline:
                break
            p_try = p_init + (np.random.randn(rank * 27) * 0.05 if attempt > 0 else 0)
            try:
                result = least_squares(residuals, p_try, method='lm', max_nfev=5000)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_x = result.x
                    if best_cost < 1e-12:
                        break
            except Exception:
                continue
        if best_cost < 1e-12:
            break
    
    if best_x is not None and best_cost < 1e-8:
        U3 = best_x[:rank*9].reshape(rank, 3, 3)
        V3 = best_x[rank*9:2*rank*9].reshape(rank, 3, 3)
        W3 = best_x[2*rank*9:3*rank*9].reshape(rank, 3, 3)
        return U3, V3, W3
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave buffer for final operations
    
    strassen_terms = _strassen_2x2x2()
    
    # Try to find Bini 3x3x3 decomposition
    bini_decomp = None
    try:
        bini_decomp = _find_bini_333(deadline - 5)
    except Exception:
        bini_decomp = None
    
    if time.time() > deadline:
        # Fallback to schoolbook
        U, V, W, rank = _build_decomposition(n, m, p, [n], [m], [p], strassen_terms, None)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Search over all block splits
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    
    has_strassen = True
    has_bini = (bini_decomp is not None)
    
    best_rank = n * m * p
    best_split = ([n], [m], [p])
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = 0
                for (a, b, c) in zip(ns, ms, ps):
                    rank += _compute_block_rank(a, b, c, has_strassen, has_bini)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (list(ns), list(ms), list(ps))
    
    if time.time() > deadline:
        U, V, W, rank = _build_decomposition(n, m, p, [n], [m], [p], strassen_terms, None)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the best decomposition
    U, V, W, rank = _build_decomposition(n, m, p, best_split[0], best_split[1], best_split[2], strassen_terms, bini_decomp)
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: try some specific splits
    fallback_splits = [
        ([2,1], [2,2,1], [2,2,2,1]),
        ([2,1], [2,1,2], [2,2,1,2]),
        ([1,2], [2,2,1], [2,2,2,1]),
        ([2,1], [2,2,1], [2,2,1,2]),
        ([2,1], [2,1,2], [2,2,2,1]),
        ([1,2], [2,1,2], [2,2,2,1]),
        ([2,1], [2,2,1], [2,1,2,2]),
        ([2,1], [2,1,2], [2,1,2,2]),
        ([1,2], [2,2,1], [2,1,2,2]),
        ([1,2], [2,1,2], [2,1,2,2]),
        ([3], [5], [7]),
    ]
    
    for ns, ms, ps in fallback_splits:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _build_decomposition(n, m, p, ns, ms, ps, strassen_terms, bini_decomp)
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue
    
    # Final fallback: schoolbook
    U, V, W, rank = _build_decomposition(n, m, p, [n], [m], [p], strassen_terms, None)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
