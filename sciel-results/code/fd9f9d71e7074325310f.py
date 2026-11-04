"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    strassen_patterns = [
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
    
    def get_compositions(d, max_part=2):
        """Generate all compositions of d into parts of size 1..max_part."""
        if d == 0:
            yield []
            return
        for first in range(1, min(max_part, d) + 1):
            for rest in get_compositions(d - first, max_part):
                yield [first] + list(rest)
    
    def compute_rank(n_splits, m_splits, p_splits):
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
    
    def build_decomposition(n_splits, m_splits, p_splits):
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
        
        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                gi = i_off + li
                                gj = j_off + lj
                                U_entries.append((gi * m + gj, col, val))
                            for lj, lk, val in pattern['V']:
                                gj = j_off + lj
                                gk = k_off + lk
                                V_entries.append((gj * p + gk, col, val))
                            for lk, li, val in pattern['W']:
                                gk = k_off + lk
                                gi = i_off + li
                                W_entries.append((gk * n + gi, col, val))
                            rank += 1
                    else:
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U_entries.append((gi * m + gj, col, 1))
                                    V_entries.append((gj * p + gk, col, 1))
                                    W_entries.append((gk * n + gi, col, 1))
                                    rank += 1
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val
        
        return {"U": U, "V": V, "W": W}
    
    # Generate all compositions
    n_comps = list(get_compositions(n))
    m_comps = list(get_compositions(m))
    p_comps = list(get_compositions(p))
    
    # Find best split combination
    best_rank = n * m * p
    best_splits = (n_comps[0], m_comps[0], p_comps[0])
    
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                r = compute_rank(nc, mc, pc)
                if r < best_rank:
                    best_rank = r
                    best_splits = (nc, mc, pc)
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    n_splits, m_splits, p_splits = best_splits
    result = build_decomposition(n_splits, m_splits, p_splits)
    
    # Verify the decomposition is correct
    import numpy as np
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    
    # Check: sum_k U[i,j,k] * V[j,l,k] * W[l,i,k] should equal delta_{i,i'} delta_{j,j'} delta_{l,l'}
    # Actually the tensor T[i,j,l] = sum_k U[i*m+j,k] * V[j*p+l,k] * W[l*n+i,k]
    # should equal 1 if (i,j,l) is a valid index and the decomposition is correct
    # For matrix multiplication: (A*B)[i,l] = sum_j A[i,j]*B[j,l]
    # So T[i,j,l] should be 1 for all valid (i,j,l)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for l in range(p):
                expected = 1.0
                actual = 0.0
                for k in range(rank):
                    actual += U[i*m+j, k] * V[j*p+l, k] * W[l*n+i, k]
                if abs(actual - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook if verification fails
        U_s = [[0] * (n*m*p) for _ in range(n*m)]
        V_s = [[0] * (n*m*p) for _ in range(m*p)]
        W_s = [[0] * (n*m*p) for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    idx = i*m*p + j*p + l
                    U_s[i*m+j][idx] = 1
                    V_s[j*p+l][idx] = 1
                    W_s[l*n+i][idx] = 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return result
# EVOLVE-BLOCK-END
