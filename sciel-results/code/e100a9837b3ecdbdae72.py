"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_start_time = time.time()
TIME_LIMIT = 145.0  # 150s wall-clock, 5s safety margin

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen patterns for 2x2x2 (rank 7 decomposition)
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
    
    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the decomposition for given splits and return (U, V, W, rank)."""
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
        
        U_entries = []
        V_entries = []
        W_entries = []
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
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
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
        
        return U, V, W, rank
    
    # Try the best known splits
    # n=3: [2,1] gives 1 block of size 2
    # m=5: [2,2,1] gives 2 blocks of size 2
    # p=5: [2,2,1] gives 2 blocks of size 2
    # This gives 4 blocks of 2x2x2, each with rank 7 (saving 1 each)
    # Total rank: 75 - 4 = 71
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    U, V, W, rank = build_decomposition(n_splits, m_splits, p_splits)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
