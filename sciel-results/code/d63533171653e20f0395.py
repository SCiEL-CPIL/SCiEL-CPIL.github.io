import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _compositions(n, max_parts=None):
    """Generate all compositions of n (ordered partitions into positive integers)."""
    if max_parts is None:
        max_parts = n
    if n == 0:
        yield []
        return
    for k in range(1, n + 1):
        for rest in _compositions(n - k, max_parts - 1):
            yield [k] + rest

def _compute_rank(n_comp, m_comp, p_comp):
    """Compute total rank for given compositions."""
    rank = 0
    for ni in n_comp:
        for mj in m_comp:
            for pk in p_comp:
                if ni == 2 and mj == 2 and pk == 2:
                    rank += 7
                else:
                    rank += ni * mj * pk
    return rank

def _find_best_splits():
    """Find the best split compositions minimizing total rank."""
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_splits = ([n], [m], [p])
    
    n_comps = list(_compositions(n))
    m_comps = list(_compositions(m))
    p_comps = list(_compositions(p))
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                r = _compute_rank(nc, mc, pc)
                if r < best_rank:
                    best_rank = r
                    best_splits = (nc, mc, pc)
    
    return best_splits, best_rank

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Uses exhaustive search over block split compositions with Strassen for 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Find optimal splits
    (n_splits, m_splits, p_splits), rank = _find_best_splits()
    
    # Compute offsets
    n_off = [0]
    for i in range(len(n_splits) - 1):
        n_off.append(n_off[-1] + n_splits[i])
    m_off = [0]
    for i in range(len(m_splits) - 1):
        m_off.append(m_off[-1] + m_splits[i])
    p_off = [0]
    for i in range(len(p_splits) - 1):
        p_off.append(p_off[-1] + p_splits[i])
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2x2
    strassen = [
        {'U': [(0,0,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (1,0,1)], 'W': [(0,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(0,1,1), (1,1,1)], 'W': [(0,0,-1), (1,1,1)]}
    ]
    
    col = 0
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                if time.time() > deadline - 5.0:
                    break
                nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                i_off = n_off[i_b]
                j_off = m_off[j_b]
                k_off = p_off[k_b]
                
                if nb == 2 and mb == 2 and pb == 2:
                    for pattern in strassen:
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
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    idx_i, idx_j, idx_k = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    idx_i = idx_i.ravel()
    idx_j = idx_j.ravel()
    idx_k = idx_k.ravel()
    
    u_rows = idx_i * m + idx_j
    v_rows = idx_j * p + idx_k
    w_rows = idx_k * n + idx_i
    
    prod = U_np[u_rows, :] * V_np[v_rows, :] * W_np[w_rows, :]
    sums = prod.sum(axis=1)
    
    # If verification fails, fall back to schoolbook
    if not np.allclose(sums, 1.0, atol=1e-9):
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

# EVOLVE-BLOCK-END
