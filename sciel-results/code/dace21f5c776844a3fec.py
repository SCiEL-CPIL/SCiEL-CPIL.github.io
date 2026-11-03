import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 5, 5
    # We expect rank 46
    # Allocate max possible rank (schoolbook 50) to be safe, then trim
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    
    current_rank = 0
    
    # Partition m=5 into [2, 2, 1]
    # Partition p=5 into [2, 2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    m_starts = [0, 2, 4]
    p_starts = [0, 2, 4]
    
    # Define Strassen terms for 2x2x2
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict (local_i, local_j) -> coeff
    # v_coeffs: dict (local_j, local_k) -> coeff
    # w_coeffs: dict (local_k, local_i) -> coeff
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00+1, C11+1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00 -> C10+1, C11-1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # M3 = A00(B10-B11) -> C01+1, C11+1
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # M4 = A11(B01-B00) -> C00+1, C10+1
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M5 = (A00+A01)B11 -> C00-1, C01+1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # M6 = (A10-A00)(B00+B01) -> C11+1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11) -> C00+1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    for j_idx, m_size in enumerate(m_splits):
        for k_idx, p_size in enumerate(p_splits):
            if time.time() > deadline:
                break
                
            m_start = m_starts[j_idx]
            p_start = p_starts[k_idx]
            
            if m_size == 2 and p_size == 2:
                # Strassen
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    if current_rank >= max_rank:
                        break
                    r = current_rank
                    for (li, lj), coeff in u_coeffs.items():
                        gi = li
                        gj = m_start + lj
                        u_idx = gi * m + gj
                        U[u_idx, r] += coeff
                    
                    for (lj, lk), coeff in v_coeffs.items():
                        gj = m_start + lj
                        gk = p_start + lk
                        v_idx = gj * p + gk
                        V[v_idx, r] += coeff
                    
                    for (lk, li), coeff in w_coeffs.items():
                        gk = p_start + lk
                        gi = li
                        w_idx = gk * n + gi
                        W[w_idx, r] += coeff
                    
                    current_rank += 1
            else:
                # Schoolbook
                # n=2, m_size, p_size
                for lj in range(m_size):
                    for lk in range(p_size):
                        for li in range(n):
                            if current_rank >= max_rank:
                                break
                            r = current_rank
                            
                            gi = li
                            gj = m_start + lj
                            gk = p_start + lk
                            
                            u_idx = gi * m + gj
                            U[u_idx, r] = 1
                            
                            v_idx = gj * p + gk
                            V[v_idx, r] = 1
                            
                            w_idx = gk * n + gi
                            W[w_idx, r] = 1
                            
                            current_rank += 1
                        if current_rank >= max_rank:
                            break
                    if current_rank >= max_rank:
                        break
            if current_rank >= max_rank:
                break
        if current_rank >= max_rank:
            break
    
    # Trim to actual rank
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
