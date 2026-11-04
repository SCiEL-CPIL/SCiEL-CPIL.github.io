import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def get_compositions(total):
        """Generate all compositions of total."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(list(current))
                return
            for i in range(1, remaining + 1):
                current.append(i)
                helper(remaining - i, current)
                current.pop()
        helper(total, [])
        return results
    
    # Precompute compositions
    comps_n = get_compositions(n)
    comps_m = get_compositions(m)
    comps_p = get_compositions(p)
    
    # Memoized min rank computation
    min_rank_cache = {}
    
    def min_rank(a, b, c):
        """Compute minimum rank for a×b times b×c matrix multiplication."""
        if (a, b, c) in min_rank_cache:
            return min_rank_cache[(a, b, c)]
        
        # Base cases
        if a == 1 and b == 1 and c == 1:
            min_rank_cache[(a, b, c)] = 1
            return 1
        if a == 2 and b == 2 and c == 2:
            min_rank_cache[(a, b, c)] = 7
            return 7
        
        # Schoolbook upper bound
        best = a * b * c
        
        # Try splitting each dimension
        for a1 in range(1, a):
            a2 = a - a1
            r1 = min_rank(a1, b, c)
            r2 = min_rank(a2, b, c)
            best = min(best, r1 + r2)
        
        for b1 in range(1, b):
            b2 = b - b1
            r1 = min_rank(a, b1, c)
            r2 = min_rank(a, b2, c)
            best = min(best, r1 + r2)
        
        for c1 in range(1, c):
            c2 = c - c1
            r1 = min_rank(a, b, c1)
            r2 = min_rank(a, b, c2)
            best = min(best, r1 + r2)
        
        min_rank_cache[(a, b, c)] = best
        return best
    
    # Search over all partitions
    best_total_rank = n * m * p  # schoolbook
    best_partition = (comps_n[0], comps_m[0], comps_p[0])
    
    for cn in comps_n:
        for cm in comps_m:
            for cp in comps_p:
                total_rank = 0
                for a in cn:
                    for b in cm:
                        for c in cp:
                            total_rank += min_rank(a, b, c)
                            if total_rank >= best_total_rank:
                                break
                        if total_rank >= best_total_rank:
                            break
                    if total_rank >= best_total_rank:
                        break
                if total_rank < best_total_rank:
                    best_total_rank = total_rank
                    best_partition = (cn, cm, cp)
    
    # Now construct the actual factor matrices
    cn, cm, cp = best_partition
    
    rank = best_total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Compute block offsets
    n_offsets = [0]
    for a in cn:
        n_offsets.append(n_offsets[-1] + a)
    m_offsets = [0]
    for b in cm:
        m_offsets.append(m_offsets[-1] + b)
    p_offsets = [0]
    for c in cp:
        p_offsets.append(p_offsets[-1] + c)
    
    for i in range(len(cn)):
        for j in range(len(cm)):
            for k in range(len(cp)):
                a = cn[i]
                b = cm[j]
                c = cp[k]
                
                if a == 2 and b == 2 and c == 2:
                    # Strassen for 2x2 x 2x2
                    # M1 = (A11-A22)*(B21-B22)
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+0)][col] += 1
                    U[(n_offsets[i]+1)*m + (m_offsets[j]+1)][col] += -1
                    V[(m_offsets[j]+1)*p + (p_offsets[k]+0)][col] += 1
                    V[(m_offsets[j]+1)*p + (p_offsets[k]+1)][col] += -1
                    W[(p_offsets[k]+0)*n + (n_offsets[i]+0)][col] += 1
                    W[(p_offsets[k]+1)*n + (n_offsets[i]+1)][col] += 1
                    col += 1
                    
                    # M2 = (A11+A12)*B22
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+0)][col] += 1
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+1)][col] += 1
                    V[(m_offsets[j]+1)*p + (p_offsets[k]+1)][col] += 1
                    W[(p_offsets[k]+1)*n + (n_offsets[i]+0)][col] += 1
                    W[(p_offsets[k]+1)*n + (n_offsets[i]+1)][col] += -1
                    col += 1
                    
                    # M3 = A11*(B11-B22)
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+0)][col] += 1
                    V[(m_offsets[j]+0)*p + (p_offsets[k]+0)][col] += 1
                    V[(m_offsets[j]+1)*p + (p_offsets[k]+1)][col] += -1
                    W[(p_offsets[k]+1)*n + (n_offsets[i]+0)][col] += 1
                    W[(p_offsets[k]+1)*n + (n_offsets[i]+1)][col] += 1
                    col += 1
                    
                    # M4 = A22*(B11-B12)
                    U[(n_offsets[i]+1)*m + (m_offsets[j]+1)][col] += 1
                    V[(m_offsets[j]+0)*p + (p_offsets[k]+0)][col] += 1
                    V[(m_offsets[j]+0)*p + (p_offsets[k]+1)][col] += -1
                    W[(p_offsets[k]+0)*n + (n_offsets[i]+0)][col] += 1
                    W[(p_offsets[k]+1)*n + (n_offsets[i]+1)][col] += 1
                    col += 1
                    
                    # M5 = (A12+A22)*(B11-B21)
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+1)][col] += 1
                    U[(n_offsets[i]+1)*m + (m_offsets[j]+1)][col] += 1
                    V[(m_offsets[j]+0)*p + (p_offsets[k]+0)][col] += 1
                    V[(m_offsets[j]+1)*p + (p_offsets[k]+0)][col] += -1
                    W[(p_offsets[k]+0)*n + (n_offsets[i]+0)][col] += -1
                    W[(p_offsets[k]+0)*n + (n_offsets[i]+1)][col] += 1
                    col += 1
                    
                    # M6 = (A11-A12)*(B12+B22)
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+0)][col] += 1
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+1)][col] += -1
                    V[(m_offsets[j]+0)*p + (p_offsets[k]+1)][col] += 1
                    V[(m_offsets[j]+1)*p + (p_offsets[k]+1)][col] += 1
                    W[(p_offsets[k]+0)*n + (n_offsets[i]+1)][col] += 1
                    col += 1
                    
                    # M7 = (A21+A11)*B11
                    U[(n_offsets[i]+1)*m + (m_offsets[j]+0)][col] += 1
                    U[(n_offsets[i]+0)*m + (m_offsets[j]+0)][col] += 1
                    V[(m_offsets[j]+0)*p + (p_offsets[k]+0)][col] += 1
                    W[(p_offsets[k]+0)*n + (n_offsets[i]+0)][col] += 1
                    W[(p_offsets[k]+0)*n + (n_offsets[i]+1)][col] += 1
                    col += 1
                else:
                    # Schoolbook for this block
                    for li in range(a):
                        for lj in range(b):
                            for lk in range(c):
                                U[(n_offsets[i]+li)*m + (m_offsets[j]+lj)][col] += 1
                                V[(m_offsets[j]+lj)*p + (p_offsets[k]+lk)][col] += 1
                                W[(p_offsets[k]+lk)*n + (n_offsets[i]+li)][col] += 1
                                col += 1
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
