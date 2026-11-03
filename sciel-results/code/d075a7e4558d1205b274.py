"""Matrix multiplication 3x5x5 - Rank-minimal exact decomposition search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_patterns():
    """
    Return the 7 Strassen patterns for 2x2x2 matrix multiplication.
    Each pattern is a dict with 'U', 'V', 'W' lists of (local_i, local_j, value) tuples.
    For U: (i, j, val) means U[i, j] += val in the local 2x2 block.
    For V: (j, k, val) means V[j, k] += val in the local 2x2 block.
    For W: (k, i, val) means W[k, i] += val in the local 2x2 block.
    """
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
    # Calculate offsets for each block
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
    
    # Initialize matrices with zeros
    # Using lists of lists for flexibility, will convert to numpy for verification if needed
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
                    # Apply Strassen patterns
                    for pattern in strassen_patterns:
                        # Fill U
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U[gi * m + gj][col] += val
                        # Fill V
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk][col] += val
                        # Fill W
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi][col] += val
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(nb):
                        for lj in range(mb):
                            for lk in range(pb):
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] += 1
                                V[gj * p + gk][col] += 1
                                W[gk * n + gi][col] += 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor identity for the decomposition."""
    # Convert to numpy arrays for faster vectorized checks if needed, 
    # but the loop is small enough.
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all non-zero entries of the target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # sum_r U[u,r] * V[v,r] * W[w,r]
                s = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
                # Also check that other entries are 0? 
                # The rank-minimal decomposition must be exact.
                # Checking all zero entries is O(nm*mp*pn*rank) which is 15*25*15*26 ~ 1.4M ops. 
                # This is fast enough.
    
    # Check zero entries
    # To be safe, we should check a few random zero entries or all if fast.
    # Given the small size, let's check all.
    # Create a mask of non-zero entries
    # Actually, iterating over all (u,v,w) and checking is 15*25*15 = 5625 checks.
    # Each check is a dot product of size rank (~26).
    # 5625 * 26 = ~150k ops. Very fast.
    
    # We already checked the 1s. Now check 0s.
    # We can do this by constructing the full tensor and checking, but that's memory heavy.
    # Better: iterate u, v, w.
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                # Determine if (u,v,w) corresponds to a 1 in T
                # u = i*m + j => i = u // m, j = u % m
                i = u // m
                j = u % m
                # v = j*p + k => k = v % p, check j == v // p
                if v // p != j:
                    # Should be 0
                    s = np.dot(U_np[u], V_np[v] * W_np[w])
                    if not np.isclose(s, 0.0, atol=1e-9):
                        return False
                else:
                    # This is a 1 entry, already checked in the first loop?
                    # No, the first loop iterated i,j,k and computed u,v,w.
                    # This loop covers all u,v,w.
                    # If it's a 1 entry, we already verified it equals 1.
                    # If it's a 0 entry, we verify it equals 0.
                    pass

    return True

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 120.0
    
    # Generate all possible splits
    n_splits_list = _get_splits(n)
    m_splits_list = _get_splits(m)
    p_splits_list = _get_splits(p)
    
    best_rank = n * m * p  # schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute schoolbook factors as fallback
    def get_schoolbook():
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
    
    school_U, school_V, school_W = get_schoolbook()
    best_U, best_V, best_W = school_U, school_V, school_W
    
    # Try all combinations of splits
    for n_splits in n_splits_list:
        for m_splits in m_splits_list:
            for p_splits in p_splits_list:
                if time.time() > deadline - 10:
                    break
                
                rank = _compute_rank(n_splits, m_splits, p_splits)
                
                if rank < best_rank:
                    U, V, W = _build_decomposition(n_splits, m_splits, p_splits, n, m, p)
                    
                    # Verify
                    if _verify_decomposition(U, V, W, n, m, p):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
# EVOLVE-BLOCK-END
