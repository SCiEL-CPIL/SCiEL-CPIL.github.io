import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
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
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U_np.shape[1]):
                    s += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W

def _try_2x2x2_block_placement(n, m, p):
    """
    Try to place a 2x2x2 Strassen block and fill the rest with schoolbook.
    Returns (rank, U, V, W) if valid, else None.
    """
    strassen_patterns = _strassen_patterns()
    rank_strassen = 7
    rank_schoolbook = n * m * p - 8  # remaining entries
    
    # Try all possible positions for the 2x2x2 block
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Check if this placement is valid (no overlap issues for now)
                # Build the decomposition
                total_rank = rank_strassen + rank_schoolbook
                
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                
                col = 0
                # Place Strassen block
                for pattern in strassen_patterns:
                    for li, lj, val in pattern['U']:
                        gi = i0 + li
                        gj = j0 + lj
                        U[gi * m + gj][col] = val
                    for lj, lk, val in pattern['V']:
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = val
                    for lk, li, val in pattern['W']:
                        gk = k0 + lk
                        gi = i0 + li
                        W[gk * n + gi][col] = val
                    col += 1
                
                # Fill rest with schoolbook
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Skip if this entry is covered by the Strassen block
                            if (i0 <= i < i0 + 2) and (j0 <= j < j0 + 2) and (k0 <= k < k0 + 2):
                                continue
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    return total_rank, U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_rank = n * m * p  # schoolbook rank = 75
    best_U, best_V, best_W = None, None, None
    
    # First, try the direct 2x2x2 + schoolbook approach
    result = _try_2x2x2_block_placement(n, m, p)
    if result is not None:
        rank, U, V, W = result
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    # Then, try exhaustive split search
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                if time.time() > deadline:
                    break
                
                rank = _compute_rank(n_splits, m_splits, p_splits)
                
                if rank < best_rank:
                    U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
                    
                    # Verify
                    if _verify_decomposition(U, V, W, n, m, p):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
    
    # If we found a better decomposition, return it
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
