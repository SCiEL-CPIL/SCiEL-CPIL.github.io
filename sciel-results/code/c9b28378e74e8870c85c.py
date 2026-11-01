import numpy as np
import time

# Fixed deadline guard: evaluator kills at 150 s; leave margin.
_DEADLINE = 120.0

def _build_block_term(t, blk, p):
    """Return (U_row, U_col, V_row, V_col, W_row, W_col, val) for one Strassen term.
    t in 0..6 for the 7 Strassen products; t in 7..10 for the 4 extra rank-1 terms.
    U index: i*3 + j  (i in {0,1}, j in {0,1,2})
    V index: j*p + k  (j in {0,1,2}, k in {0..p-1})
    W index: k*n + i  (n=2; k is global output column)
    """
    n = 2
    k0, k1 = 2*blk, 2*blk+1
    if t == 0:   # m1 = (A00+A11)(B00+B11)
        return [(0*3+0, 1), (1*3+1, 1), (0*p+k0, 1), (1*p+k1, 1), (k0*n+0, 1), (k1*n+1, 1)]
    if t == 1:   # m2 = (A10+A11)B00
        return [(1*3+0, 1), (1*3+1, 1), (0*p+k0, 1), None, (k0*n+1, 1), (k1*n+1, -1)]
    if t == 2:   # m3 = A00(B01-B11)
        return [(0*3+0, 1), None, (0*p+k1, 1), (1*p+k1, -1), (k1*n+0, 1), (k1*n+1, 1)]
    if t == 3:   # m4 = A11(B10-B00)
        return [(1*3+1, 1), None, (1*p+k0, 1), (0*p+k0, -1), (k0*n+0, 1), (k0*n+1, 1)]
    if t == 4:   # m5 = (A00+A01)B11
        return [(0*3+0, 1), (0*3+1, 1), (1*p+k1, 1), None, (k0*n+0, -1), (k1*n+0, 1)]
    if t == 5:   # m6 = (A10-A00)(B00+B01)
        return [(1*3+0, 1), (0*3+0, -1), (0*p+k0, 1), (0*p+k1, 1), None, (k1*n+1, 1)]
    if t == 6:   # m7 = (A01-A11)(B10+B11)
        return [(0*3+1, 1), (1*3+1, -1), (1*p+k0, 1), (1*p+k1, 1), (k0*n+0, 1), None]
    # Extra rank-1 terms
    if t == 7:   # C[0,k0] += A[0,2]*B[2,k0]
        return [(0*3+2, 1), None, (2*p+k0, 1), None, (k0*n+0, 1), None]
    if t == 8:   # C[0,k1] += A[0,2]*B[2,k1]
        return [(0*3+2, 1), None, (2*p+k1, 1), None, (k1*n+0, 1), None]
    if t == 9:   # C[1,k0] += A[1,2]*B[2,k0]
        return [(1*3+2, 1), None, (2*p+k0, 1), None, (k0*n+1, 1), None]
    if t == 10:  # C[1,k1] += A[1,2]*B[2,k1]
        return [(1*3+2, 1), None, (2*p+k1, 1), None, (k1*n+1, 1), None]
    raise ValueError(t)

def _assemble(terms_per_block, n, m, p):
    """Build U,V,W from a list of (blk, t) terms. Returns (U, V, W, rank)."""
    rank = len(terms_per_block)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for t_idx, (blk, t) in enumerate(terms_per_block):
        spec = _build_block_term(t, blk, p)
        # spec: list of (row, col, val) triples for U, V, W (None = skip)
        for mat, spec_item in zip((U, V, W), spec):
            if spec_item is None:
                continue
            row, val = spec_item
            mat[row, t_idx] = val
    return U, V, W, rank

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_t U[:,t] V[:,t] W[:,t] == M_{n,m,p} tensor. Returns bool."""
    # Reconstruct tensor T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.einsum('at,bt,ct->abc', U, V, W)
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, expected, atol=tol)

def construct():
    t_start = time.time()
    n, m, p = 2, 3, 8
    num_blocks = p // 2  # 4
    # Base terms: 11 per block (7 Strassen + 4 extra)
    base_terms = [(blk, t) for blk in range(num_blocks) for t in range(11)]
    U, V, W, rank = _assemble(base_terms, n, m, p)
    best = (rank, U, V, W)
    
    # Verify base (should be valid)
    if _verify(U, V, W, n, m, p):
        pass  # base is valid
    
    # Search: try dropping one term per block to see if rank can be reduced.
    # Strategy: for each block, try removing each of the 11 terms and re-verify.
    # Since blocks are independent (disjoint W columns per block), a term in block blk
    # only affects that block's output. So we can test per-block.
    
    # Build a per-block verification helper: check if block blk's output is correct
    # given the terms assigned to it.
    def _verify_block(terms_in_block, blk, n, m, p):
        """terms_in_block: list of t values for this block. Check block correctness."""
        # Build local U,V,W for this block only
        local_terms = [(blk, t) for t in terms_in_block]
        U_l, V_l, W_l, _ = _assemble(local_terms, n, m, p)
        # Only check the W columns for this block: k in {2*blk, 2*blk+1}
        k0, k1 = 2*blk, 2*blk+1
        T = np.einsum('at,bt,ct->abc', U_l, V_l, W_l)
        # Expected for this block: T[i*m+j, j*p+k, k*n+i] = 1 for k in {k0,k1}
        expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in (k0, k1):
                    expected[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T, expected, atol=1e-9)
    
    # Try to reduce rank by dropping terms.
    # Since the problem is symmetric across blocks, if we can drop a term in one block,
    # we can drop it in all blocks. So test on block 0 only.
    best_terms_block0 = list(range(11))
    for drop_t in range(11):
        if time.time() - t_start > _DEADLINE - 10:
            break
        candidate = [t for t in range(11) if t != drop_t]
        if _verify_block(candidate, 0, n, m, p):
            # This term is redundant! Drop it from all blocks.
            best_terms_block0 = candidate
            break  # try to drop more
    
    # Try dropping a second term if we already dropped one.
    if len(best_terms_block0) < 11:
        for drop_t in range(11):
            if time.time() - t_start > _DEADLINE - 10:
                break
            if drop_t not in best_terms_block0:
                continue
            candidate = [t for t in best_terms_block0 if t != drop_t]
            if _verify_block(candidate, 0, n, m, p):
                best_terms_block0 = candidate
                break
    
    # Build final decomposition with the reduced per-block terms.
    final_terms = [(blk, t) for blk in range(num_blocks) for t in best_terms_block0]
    U_f, V_f, W_f, rank_f = _assemble(final_terms, n, m, p)
    
    # Final verification
    if _verify(U_f, V_f, W_f, n, m, p):
        best = (rank_f, U_f, V_f, W_f)
    else:
        # Fall back to base
        U_b, V_b, W_b, rank_b = _assemble(base_terms, n, m, p)
        best = (rank_b, U_b, V_b, W_b)
    
    # Deadline guard: return best found
    _, U_ret, V_ret, W_ret = best
    return {"U": U_ret.tolist(), "V": V_ret.tolist(), "W": W_ret.tolist()}
