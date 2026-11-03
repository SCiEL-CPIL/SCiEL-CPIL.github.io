import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    start_time = time.time()
    deadline = start_time + 100
    
    def strassen_2x2x2():
        U_s = np.zeros((4, 7), dtype=float)
        V_s = np.zeros((4, 7), dtype=float)
        W_s = np.zeros((4, 7), dtype=float)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = -1; W_s[2, 3] = 1
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = 1; W_s[1, 4] = 1
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = -1
        return U_s, V_s, W_s
    
    def schoolbook_2x2x1():
        U_s = np.zeros((4, 4), dtype=float)
        V_s = np.zeros((4, 4), dtype=float)
        W_s = np.zeros((4, 4), dtype=float)
        U_s[0, 0] = 1; V_s[0, 0] = 1; W_s[0, 0] = 1
        U_s[1, 1] = 1; V_s[1, 1] = 1; W_s[0, 1] = 1
        U_s[2, 2] = 1; V_s[0, 2] = 1; W_s[1, 2] = 1
        U_s[3, 3] = 1; V_s[1, 3] = 1; W_s[1, 3] = 1
        return U_s, V_s, W_s
    
    def build_decomposition(m_split, p_split):
        """Build decomposition with given m and p splits.
        m_split: list of block sizes summing to m
        p_split: list of block sizes summing to p
        """
        # For each m-block of size mb and p-block of size pb:
        # If mb==2 and pb==2: use Strassen, rank 7
        # If mb==2 and pb==1: use schoolbook 2x2x1, rank 4
        # If mb==1 and pb==anything: use schoolbook, rank 1*1*pb = pb
        # If mb==2 and pb>2: split further or use schoolbook rank 2*2*pb
        
        total_rank = 0
        for mb in m_split:
            for pb in p_split:
                if mb == 2 and pb == 2:
                    total_rank += 7
                elif mb == 2 and pb == 1:
                    total_rank += 4
                elif mb == 1:
                    total_rank += pb  # 1x1 x pb, rank pb
                else:
                    total_rank += mb * pb  # schoolbook fallback
        
        if total_rank >= 50:
            return None, total_rank
        
        # Build the actual matrices
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        m_offset = 0
        for mb in m_split:
            p_offset = 0
            for pb in p_split:
                if mb == 2 and pb == 2:
                    sU, sV, sW = strassen_2x2x2()
                    for t in range(7):
                        for i in range(2):
                            for jl in range(2):
                                ag = i * m + (m_offset + jl)
                                U[ag, col + t] += sU[i * 2 + jl, t]
                        for jl in range(2):
                            for kl in range(2):
                                bg = (m_offset + jl) * p + (p_offset + kl)
                                V[bg, col + t] += sV[jl * 2 + kl, t]
                        for kl in range(2):
                            for i in range(2):
                                cg = (p_offset + kl) * n + i
                                W[cg, col + t] += sW[kl * 2 + i, t]
                    col += 7
                elif mb == 2 and pb == 1:
                    scU, scV, scW = schoolbook_2x2x1()
                    for t in range(4):
                        for i in range(2):
                            for jl in range(2):
                                ag = i * m + (m_offset + jl)
                                U[ag, col + t] += scU[i * 2 + jl, t]
                        V[(m_offset) * p + p_offset, col + t] += scV[0, t]
                        V[(m_offset + 1) * p + p_offset, col + t] += scV[1, t]
                        W[p_offset * n + 0, col + t] += scW[0, t]
                        W[p_offset * n + 1, col + t] += scW[1, t]
                    col += 4
                elif mb == 1:
                    # 1x1 x pb: each is A[i, j] * B[j, k] -> C[i, k]
                    # For each k in the pb block, we need 2 products (one for i=0, one for i=1)
                    for kl in range(pb):
                        for i in range(2):
                            U[i * m + m_offset, col] = 1.0
                            V[(m_offset) * p + (p_offset + kl), col] = 1.0
                            W[(p_offset + kl) * n + i, col] = 1.0
                            col += 1
                else:
                    # Schoolbook fallback
                    for i in range(2):
                        for jl in range(mb):
                            for kl in range(pb):
                                U[i * m + (m_offset + jl), col] = 1.0
                                V[(m_offset + jl) * p + (p_offset + kl), col] = 1.0
                                W[(p_offset + kl) * n + i, col] = 1.0
                                col += 1
                p_offset += pb
            m_offset += mb
        
        if col != total_rank:
            return None, total_rank
        
        return (U, V, W), total_rank
    
    # Try various splits
    best_result = None
    best_rank = 56
    
    # Current best: m_split=[2,2], p_split=[2,2,2,1] -> rank 50
    splits_to_try = [
        ([2, 2], [2, 2, 2, 1]),  # rank 50
        ([2, 2], [2, 2, 1, 2]),  # same, different order
        ([2, 2], [2, 1, 2, 2]),
        ([2, 2], [1, 2, 2, 2]),
        ([1, 3], [2, 2, 2, 1]),
        ([3, 1], [2, 2, 2, 1]),
        ([1, 1, 2], [2, 2, 2, 1]),
        ([2, 1, 1], [2, 2, 2, 1]),
        ([1, 1, 1, 1], [2, 2, 2, 1]),
    ]
    
    for m_split, p_split in splits_to_try:
        if time.time() > deadline - 10:
            break
        if sum(m_split) != m or sum(p_split) != p:
            continue
        result, rank = build_decomposition(m_split, p_split)
        if result is not None and rank < best_rank:
            best_rank = rank
            best_result = result
    
    if best_result is None:
        # Fallback to rank 50
        result, _ = build_decomposition([2, 2], [2, 2, 2, 1])
        best_result = result
        best_rank = 50
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
