import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222_terms():
    """
    Returns a list of 7 dicts, each with keys 'U', 'V', 'W'.
    Each value is a list of (local_i, local_j, coeff) tuples.
    U is indexed (i, j), V is indexed (j, k), W is indexed (k, i).
    """
    return [
        # M1: (A00+A11)(B00+B11)
        {'U': [(0,0,1), (1,1,1)],
         'V': [(0,0,1), (1,1,1)],
         'W': [(0,0,1), (1,1,1)]},
        # M2: (A10+A11)B00
        {'U': [(1,0,1), (1,1,1)],
         'V': [(0,0,1)],
         'W': [(0,1,1), (1,1,-1)]},
        # M3: A00(B01-B11)
        {'U': [(0,0,1)],
         'V': [(0,1,1), (1,1,-1)],
         'W': [(1,0,1), (1,1,1)]},
        # M4: A11(B10-B00)
        {'U': [(1,1,1)],
         'V': [(1,0,1), (0,0,-1)],
         'W': [(0,0,1), (0,1,1)]},
        # M5: (A00+A01)B11
        {'U': [(0,0,1), (0,1,1)],
         'V': [(1,1,1)],
         'W': [(0,0,-1), (1,0,1)]},
        # M6: (A10-A00)(B00+B01)
        {'U': [(1,0,1), (0,0,-1)],
         'V': [(0,0,1), (0,1,1)],
         'W': [(1,1,1)]},
        # M7: (A01-A11)(B10+B11)
        {'U': [(0,1,1), (1,1,-1)],
         'V': [(1,0,1), (1,1,1)],
         'W': [(0,0,1)]}
    ]

def _build_decomposition(n_splits, m_splits, p_splits):
    """
    Builds the tensor decomposition using block partitioning.
    Uses Strassen for 2x2x2 blocks, schoolbook otherwise.
    Returns (U, V, W, rank)
    """
    n, m, p = DIMENSIONS
    
    # Calculate offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    # Pre-compute Strassen patterns
    strassen_patterns = _strassen_222_terms()
    
    # Estimate max rank (schoolbook)
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    rank = 0
    
    for i_b in range(len(n_splits)):
        for j_b in range(len(m_splits)):
            for k_b in range(len(p_splits)):
                n_b = n_splits[i_b]
                m_b = m_splits[j_b]
                p_b = p_splits[k_b]
                
                i_off = n_offsets[i_b]
                j_off = m_offsets[j_b]
                k_off = p_offsets[k_b]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Use Strassen
                    for pattern in strassen_patterns:
                        col = rank
                        # U
                        for li, lj, val in pattern['U']:
                            gi = i_off + li
                            gj = j_off + lj
                            U[gi * m + gj][col] = val
                        # V
                        for lj, lk, val in pattern['V']:
                            gj = j_off + lj
                            gk = k_off + lk
                            V[gj * p + gk][col] = val
                        # W
                        for lk, li, val in pattern['W']:
                            gk = k_off + lk
                            gi = i_off + li
                            W[gk * n + gi][col] = val
                        rank += 1
                else:
                    # Schoolbook
                    for li in range(n_b):
                        for lj in range(m_b):
                            for lk in range(p_b):
                                col = rank
                                gi = i_off + li
                                gj = j_off + lj
                                gk = k_off + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                rank += 1
    
    # Trim matrices to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _estimate_rank(n_splits, m_splits, p_splits):
    """Estimate the rank for a given partitioning."""
    total_rank = 0
    for n_b in n_splits:
        for m_b in m_splits:
            for p_b in p_splits:
                if n_b == 2 and m_b == 2 and p_b == 2:
                    total_rank += 7
                else:
                    total_rank += n_b * m_b * p_b
    return total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Generate all valid partitions for each dimension
    # n=3: [2,1], [1,2]
    # m=5: [2,2,1], [2,1,2], [1,2,2], [3,2], [2,3], [4,1], [1,4], [5], [2,2,1] variants
    # p=5: same as m
    
    n_options = [[2, 1], [1, 2]]
    m_options = [[2, 2, 1], [2, 1, 2], [1, 2, 2], [3, 2], [2, 3], [4, 1], [1, 4], [5]]
    p_options = [[2, 2, 1], [2, 1, 2], [1, 2, 2], [3, 2], [2, 3], [4, 1], [1, 4], [5]]
    
    # Filter partitions that sum correctly
    n_options = [p for p in n_options if sum(p) == n]
    m_options = [p for p in m_options if sum(p) == m]
    p_options = [p for p in p_options if sum(p) == p]
    
    # Find the partition with minimum estimated rank
    best_rank = float('inf')
    best_partition = None
    
    for n_s in n_options:
        for m_s in m_options:
            for p_s in p_options:
                r = _estimate_rank(n_s, m_s, p_s)
                if r < best_rank:
                    best_rank = r
                    best_partition = (n_s, m_s, p_s)
    
    # Build the decomposition with the best partition
    if best_partition is not None:
        n_s, m_s, p_s = best_partition
        U, V, W, rank = _build_decomposition(n_s, m_s, p_s)
        
        # Verify correctness
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
