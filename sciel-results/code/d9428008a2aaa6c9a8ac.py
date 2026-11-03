"""Candidate for matrix multiplication tensor decomposition (3x5x5)."""
import time
import numpy as np

_DIM = (3, 5, 5)

def _verify(U, V, W, n, m, p):
    """Check that the decomposition is valid."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    r = len(W[0])
    
    # For each output (i,k), the bilinear form coefficient must match
    # T[i, j, k] = 1 if j is the inner index, i is row of A, k is col of B
    # Sum_r U[i,j,r] V[j,k,r] W[i,k,r] == T[i,j,k]
    # T[i,j,k] is 1 if it's a valid term in C[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 for all i, j, k? 
    # No, the tensor is defined such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k.
    # Wait, the verify in the prompt does:
    # expected = 1.0 if (a == i and b == c2 and d == k) else 0.0
    # This checks if the term A[a,b] B[c2,d] contributes to C[i,k].
    # It contributes if a=i, d=k, b=c2.
    # So for a fixed i, k, we sum over b (which is c2).
    # The verify loop iterates a, b, c2, d.
    # This is effectively checking the full tensor identity.
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c2 in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c2 and d == k) else 0.0
                            actual = 0.0
                            for rr in range(r):
                                actual += Wa[k*n+i, rr] * Ua[a*m+b, rr] * Va[c2*p+d, rr]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _strassen_patterns():
    """Return the 7 Strassen patterns for 2x2x2 matrix multiplication."""
    return [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]

def _get_splits(d):
    """Generate all compositions of d into parts of size 1 or 2."""
    results = []
    def _gen(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        if remaining >= 1:
            current.append(1)
            _gen(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _gen(remaining - 2, current)
            current.pop()
    _gen(d, [])
    return results

def _compute_rank(n_splits, m_splits, p_splits):
    """Compute the rank for given splits."""
    rank = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                if nb == 2 and mb == 2 and pb == 2:
                    rank += 7
                else:
                    rank += nb * mb * pb
    return rank

def _build_decomposition(n_splits, m_splits, p_splits, n, m, p):
    """Build the full decomposition for given splits."""
    n_off = [0]
    for s in n_splits[:-1]:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in m_splits[:-1]:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in p_splits[:-1]:
        p_off.append(p_off[-1] + s)
    
    rank = _compute_rank(n_splits, m_splits, p_splits)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_patterns()
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    for pattern in strassen_patterns:
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U[gi * m + gj][col] = val
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk][col] = val
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi][col] = val
                        col += 1
                else:
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = _DIM
    t0 = time.time()
    deadline = t0 + 110  # Safety margin
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank
    best_result = None
    
    # Try all combinations of splits
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                if time.time() > deadline:
                    break
                
                rank = _compute_rank(n_splits, m_splits, p_splits)
                
                if rank < best_rank:
                    best_rank = rank
                    # We can build and verify here, or just store the splits
                    # Building is fast, verification is the bottleneck
                    result = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best_result = result
    
    # If we found a better decomposition, return it
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
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
