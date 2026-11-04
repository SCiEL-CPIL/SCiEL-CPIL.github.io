"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    # Each pattern: U[i,j], V[j,k], W[k,i]
    strassen_patterns = [
        # M1 = (a+d)(e+h)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2 = (c+d)e
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1)]
        },
        # M3 = a(f-h)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1)]
        },
        # M4 = d(g-e)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,1,1)]
        },
        # M5 = (a+b)h
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        # M6 = (c-a)(e+f)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7 = (b-d)(g+h)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    def get_splits(d):
        """Generate all compositions of d into parts of size 1 or 2."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(current[:])
                return
            if remaining >= 1:
                current.append(1)
                helper(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                helper(remaining - 2, current)
                current.pop()
        helper(d, [])
        return results
    
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
                                U_entries.append(((i_off + li) * m + (j_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                V_entries.append(((j_off + lj) * p + (k_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                W_entries.append(((k_off + lk) * n + (i_off + li), col, val))
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
        
        # Build matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val
        
        return {"U": U, "V": V, "W": W}, rank
    
    # Generate all possible splits for each dimension
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)
    
    # Find the best combination of splits (minimum rank)
    best_rank = float('inf')
    best_splits = None
    best_result = None
    
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_splits = (ns, ms, ps)
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we haven't found a result yet, use the default split
    if best_splits is None:
        best_splits = ([2, 1], [2, 2, 1], [2, 2, 1])
        best_rank = compute_rank(*best_splits)
    
    # Build the decomposition for the best splits
    if time.time() < deadline:
        best_result, _ = build_decomposition(*best_splits)
    
    # Fallback to a simple schoolbook decomposition if something went wrong
    if best_result is None:
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
        
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
