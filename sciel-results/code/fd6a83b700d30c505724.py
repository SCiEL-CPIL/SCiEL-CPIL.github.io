"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

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
    
    def build_2x3x2_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build rank-11 decomposition for 2x3x2 block.
        A is 2x3, B is 3x2, C is 2x2.
        Split the middle dimension (size 3) into 2+1:
        - 2x2x2 part: Strassen (rank 7)
        - 2x1x2 part: schoolbook (rank 2)
        - 1x2x2 part: schoolbook (rank 2)
        Total: 7 + 2 + 2 = 11
        """
        result_U = []
        result_V = []
        result_W = []
        col = 0
        
        # Part 1: 2x2x2 (rows 0-1 of A, cols 0-1 of B) - Strassen
        for pattern in strassen_patterns:
            for li, lj, val in pattern['U']:
                result_U.append(((i_off + li) * m + (j_off + lj), col, val))
            for lj, lk, val in pattern['V']:
                result_V.append(((j_off + lj) * p + (k_off + lk), col, val))
            for lk, li, val in pattern['W']:
                result_W.append(((k_off + lk) * n + (i_off + li), col, val))
            col += 1
        
        # Part 2: 2x1x2 (rows 0-1 of A, col 2 of B) - schoolbook
        for li in range(2):
            for lk in range(2):
                result_U.append(((i_off + li) * m + (j_off + 2), col, 1))
                result_V.append(((j_off + 2) * p + (k_off + lk), col, 1))
                result_W.append(((k_off + lk) * n + (i_off + li), col, 1))
                col += 1
        
        # Part 3: 1x2x2 (row 2 of A, cols 0-1 of B) - schoolbook
        for lj in range(2):
            for lk in range(2):
                result_U.append(((i_off + 2) * m + (j_off + lj), col, 1))
                result_V.append(((j_off + lj) * p + (k_off + lk), col, 1))
                result_W.append(((k_off + lk) * n + (i_off + 2), col, 1))
                col += 1
        
        return result_U, result_V, result_W, col
    
    def build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build rank-11 decomposition for 2x2x3 block.
        A is 2x2, B is 2x3, C is 2x3.
        Split the last dimension (size 3) into 2+1:
        - 2x2x2 part: Strassen (rank 7)
        - 2x2x1 part: schoolbook (rank 4)
        Total: 7 + 4 = 11
        """
        result_U = []
        result_V = []
        result_W = []
        col = 0
        
        # Part 1: 2x2x2 (cols 0-1 of B) - Strassen
        for pattern in strassen_patterns:
            for li, lj, val in pattern['U']:
                result_U.append(((i_off + li) * m + (j_off + lj), col, val))
            for lj, lk, val in pattern['V']:
                result_V.append(((j_off + lj) * p + (k_off + lk), col, val))
            for lk, li, val in pattern['W']:
                result_W.append(((k_off + lk) * n + (i_off + li), col, val))
            col += 1
        
        # Part 2: 2x2x1 (col 2 of B) - schoolbook
        for li in range(2):
            for lj in range(2):
                result_U.append(((i_off + li) * m + (j_off + lj), col, 1))
                result_V.append(((j_off + lj) * p + (k_off + 2), col, 1))
                result_W.append(((k_off + 2) * n + (i_off + li), col, 1))
                col += 1
        
        return result_U, result_V, result_W, col
    
    def build_3x2x2_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build rank-11 decomposition for 3x2x2 block.
        A is 3x2, B is 2x2, C is 3x2.
        Split the first dimension (size 3) into 2+1:
        - 2x2x2 part: Strassen (rank 7)
        - 1x2x2 part: schoolbook (rank 4)
        Total: 7 + 4 = 11
        """
        result_U = []
        result_V = []
        result_W = []
        col = 0
        
        # Part 1: 2x2x2 (rows 0-1 of A) - Strassen
        for pattern in strassen_patterns:
            for li, lj, val in pattern['U']:
                result_U.append(((i_off + li) * m + (j_off + lj), col, val))
            for lj, lk, val in pattern['V']:
                result_V.append(((j_off + lj) * p + (k_off + lk), col, val))
            for lk, li, val in pattern['W']:
                result_W.append(((k_off + lk) * n + (i_off + li), col, val))
            col += 1
        
        # Part 2: 1x2x2 (row 2 of A) - schoolbook
        for lj in range(2):
            for lk in range(2):
                result_U.append(((i_off + 2) * m + (j_off + lj), col, 1))
                result_V.append(((j_off + lj) * p + (k_off + lk), col, 1))
                result_W.append(((k_off + lk) * n + (i_off + 2), col, 1))
                col += 1
        
        return result_U, result_V, result_W, col
    
    def build_3x3x3_decomposition(i_off, j_off, k_off, n, m, p):
        """
        Build decomposition for 3x3x3 block.
        Split 3 = 2+1 in each dimension, using optimized sub-blocks:
        - 2x2x2: Strassen rank 7
        - 2x2x1: schoolbook rank 4
        - 2x1x2: schoolbook rank 4
        - 1x2x2: schoolbook rank 4
        - 2x1x1: schoolbook rank 2
        - 1x2x1: schoolbook rank 2
        - 1x1x2: schoolbook rank 2
        - 1x1x1: schoolbook rank 1
        Total: 7+4+4+4+2+2+2+1 = 26
        """
        result_U = []
        result_V = []
        result_W = []
        col = 0
        
        # Splits: [2, 1] for each dimension
        n_splits = [2, 1]
        m_splits = [2, 1]
        p_splits = [2, 1]
        
        n_offsets = [0, 2]
        m_offsets = [0, 2]
        p_offsets = [0, 2]
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb = n_splits[i_b]
                    mb = m_splits[j_b]
                    pb = p_splits[k_b]
                    i_off_local = n_offsets[i_b]
                    j_off_local = m_offsets[j_b]
                    k_off_local = p_offsets[k_b]
                    
                    gi_off = i_off + i_off_local
                    gj_off = j_off + j_off_local
                    gk_off = k_off + k_off_local
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            for li, lj, val in pattern['U']:
                                result_U.append(((gi_off + li) * m + (gj_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                result_V.append(((gj_off + lj) * p + (gk_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                result_W.append(((gk_off + lk) * n + (gi_off + li), col, val))
                            col += 1
                    elif nb == 2 and mb == 2 and pb == 1:
                        # 2x2x1: can be done in rank 4 (schoolbook)
                        for li in range(2):
                            for lj in range(2):
                                result_U.append(((gi_off + li) * m + (gj_off + lj), col, 1))
                                result_V.append(((gj_off + lj) * p + gk_off, col, 1))
                                result_W.append((gk_off * n + gi_off + li, col, 1))
                                col += 1
                    elif nb == 2 and mb == 1 and pb == 2:
                        # 2x1x2: rank 4
                        for li in range(2):
                            for lk in range(2):
                                result_U.append(((gi_off + li) * m + gj_off, col, 1))
                                result_V.append((gj_off * p + gk_off + lk, col, 1))
                                result_W.append(((gk_off + lk) * n + gi_off + li, col, 1))
                                col += 1
                    elif nb == 1 and mb == 2 and pb == 2:
                        # 1x2x2: rank 4
                        for lj in range(2):
                            for lk in range(2):
                                result_U.append(((gi_off) * m + gj_off + lj, col, 1))
                                result_V.append(((gj_off + lj) * p + gk_off + lk, col, 1))
                                result_W.append(((gk_off + lk) * n + gi_off, col, 1))
                                col += 1
                    elif nb == 2 and mb == 1 and pb == 1:
                        # 2x1x1: rank 2
                        for li in range(2):
                            result_U.append(((gi_off + li) * m + gj_off, col, 1))
                            result_V.append((gj_off * p + gk_off, col, 1))
                            result_W.append((gk_off * n + gi_off + li, col, 1))
                            col += 1
                    elif nb == 1 and mb == 2 and pb == 1:
                        # 1x2x1: rank 2
                        for lj in range(2):
                            result_U.append((gi_off * m + gj_off + lj, col, 1))
                            result_V.append(((gj_off + lj) * p + gk_off, col, 1))
                            result_W.append((gk_off * n + gi_off, col, 1))
                            col += 1
                    elif nb == 1 and mb == 1 and pb == 2:
                        # 1x1x2: rank 2
                        for lk in range(2):
                            result_U.append((gi_off * m + gj_off, col, 1))
                            result_V.append((gj_off * p + gk_off + lk, col, 1))
                            result_W.append(((gk_off + lk) * n + gi_off, col, 1))
                            col += 1
                    elif nb == 1 and mb == 1 and pb == 1:
                        # 1x1x1: rank 1
                        result_U.append((gi_off * m + gj_off, col, 1))
                        result_V.append((gj_off * p + gk_off, col, 1))
                        result_W.append((gk_off * n + gi_off, col, 1))
                        col += 1
        
        return result_U, result_V, result_W, col
    
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
                    nb = n_splits[i_b]
                    mb = m_splits[j_b]
                    pb = p_splits[k_b]
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
                    elif (nb == 2 and mb == 3 and pb == 2) or \
                         (nb == 3 and mb == 2 and pb == 2) or \
                         (nb == 2 and mb == 2 and pb == 3) or \
                         (nb == 3 and mb == 3 and pb == 2) or \
                         (nb == 3 and mb == 2 and pb == 3) or \
                         (nb == 2 and mb == 3 and pb == 3) or \
                         (nb == 3 and mb == 3 and pb == 3):
                        # Use optimized decomposition
                        if nb == 2 and mb == 3 and pb == 2:
                            u_ent, v_ent, w_ent, r = build_2x3x2_decomposition(i_off, j_off, k_off, n, m, p)
                        elif nb == 3 and mb == 2 and pb == 2:
                            u_ent, v_ent, w_ent, r = build_3x2x2_decomposition(i_off, j_off, k_off, n, m, p)
                        elif nb == 2 and mb == 2 and pb == 3:
                            u_ent, v_ent, w_ent, r = build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p)
                        elif nb == 3 and mb == 3 and pb == 3:
                            u_ent, v_ent, w_ent, r = build_3x3x3_decomposition(i_off, j_off, k_off, n, m, p)
                        elif nb == 3 and mb == 3 and pb == 2:
                            # 3x3x2: split first dim into 2+1
                            # 2x3x2 (rank 11) + 1x3x2 (rank 6) = 17
                            u_ent, v_ent, w_ent, r1 = build_2x3x2_decomposition(i_off, j_off, k_off, n, m, p)
                            u_ent2, v_ent2, w_ent2, r2 = [], [], [], 0
                            for lj in range(3):
                                for lk in range(2):
                                    u_ent2.append(((i_off + 2) * m + (j_off + lj), r2, 1))
                                    v_ent2.append(((j_off + lj) * p + (k_off + lk), r2, 1))
                                    w_ent2.append(((k_off + lk) * n + (i_off + 2), r2, 1))
                                    r2 += 1
                            u_ent = u_ent + u_ent2
                            v_ent = v_ent + v_ent2
                            w_ent = w_ent + w_ent2
                            r = r1 + r2
                        elif nb == 3 and mb == 2 and pb == 3:
                            # 3x2x3: split first dim into 2+1
                            # 2x2x3 (rank 11) + 1x2x3 (rank 6) = 17
                            u_ent, v_ent, w_ent, r1 = build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p)
                            u_ent2, v_ent2, w_ent2, r2 = [], [], [], 0
                            for lj in range(2):
                                for lk in range(3):
                                    u_ent2.append(((i_off + 2) * m + (j_off + lj), r2, 1))
                                    v_ent2.append(((j_off + lj) * p + (k_off + lk), r2, 1))
                                    w_ent2.append(((k_off + lk) * n + (i_off + 2), r2, 1))
                                    r2 += 1
                            u_ent = u_ent + u_ent2
                            v_ent = v_ent + v_ent2
                            w_ent = w_ent + w_ent2
                            r = r1 + r2
                        elif nb == 2 and mb == 3 and pb == 3:
                            # 2x3x3: split second dim into 2+1
                            # 2x2x3 (rank 11) + 2x1x3 (rank 6) = 17
                            u_ent, v_ent, w_ent, r1 = build_2x2x3_decomposition(i_off, j_off, k_off, n, m, p)
                            u_ent2, v_ent2, w_ent2, r2 = [], [], [], 0
                            for li in range(2):
                                for lk in range(3):
                                    u_ent2.append(((i_off + li) * m + (j_off + 2), r2, 1))
                                    v_ent2.append(((j_off + 2) * p + (k_off + lk), r2, 1))
                                    w_ent2.append(((k_off + lk) * n + (i_off + li), r2, 1))
                                    r2 += 1
                            u_ent = u_ent + u_ent2
                            v_ent = v_ent + v_ent2
                            w_ent = w_ent + w_ent2
                            r = r1 + r2
                        else:
                            # Fallback to schoolbook
                            u_ent, v_ent, w_ent, r = [], [], [], 0
                            for li in range(nb):
                                for lj in range(mb):
                                    for lk in range(pb):
                                        u_ent.append(((i_off + li) * m + (j_off + lj), r, 1))
                                        v_ent.append(((j_off + lj) * p + (k_off + lk), r, 1))
                                        w_ent.append(((k_off + lk) * n + (i_off + li), r, 1))
                                        r += 1
                        
                        for e in u_ent:
                            U_entries.append((e[0], rank + e[1], e[2]))
                        for e in v_ent:
                            V_entries.append((e[0], rank + e[1], e[2]))
                        for e in w_ent:
                            W_entries.append((e[0], rank + e[1], e[2]))
                        rank += r
                    else:
                        # Schoolbook
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
    
    # Generate all possible splits for each dimension (parts of size 1 or 2)
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)
    
    # Find the best combination of splits (minimum rank)
    best_rank = float('inf')
    best_result = None
    best_splits = None
    
    # Sort split options by estimated cost to find good solutions early
    def est_rank(ns, ms, ps):
        r = 0
        n_off = [0]
        for s in ns[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in ms[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in ps[:-1]:
            p_off.append(p_off[-1] + s)
        for i_b in range(len(ns)):
            for j_b in range(len(ms)):
                for k_b in range(len(ps)):
                    nb, mb, pb = ns[i_b], ms[j_b], ps[k_b]
                    if nb == 2 and mb == 2 and pb == 2:
                        r += 7
                    else:
                        r += nb * mb * pb
        return r
    
    all_combos = [(ns, ms, ps) for ns in n_split_options for ms in m_split_options for ps in p_split_options]
    all_combos.sort(key=lambda x: est_rank(*x))
    
    for ns, ms, ps in all_combos:
        if time.time() > deadline:
            break
        result, r = build_decomposition(ns, ms, ps)
        if r < best_rank:
            best_rank = r
            best_result = result
            best_splits = (ns, ms, ps)
    
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
