import time
_start_time = time.time()
_TIME_LIMIT = 110  # Safety margin before 120s limit

DIMENSIONS = [5, 5, 5]

def _get_partitions(d):
    """Generate all integer partitions of d."""
    def _partitions(n, max_part):
        if n == 0:
            yield []
            return
        for i in range(min(max_part, n), 0, -1):
            for rest in _partitions(n - i, i):
                yield [i] + rest
    return list(_partitions(d, d))

def _compute_rank(n_parts, m_parts, p_parts):
    """Compute total rank for given partitions."""
    total = 0
    for ni in n_parts:
        for mj in m_parts:
            for pk in p_parts:
                if ni == 2 and mj == 2 and pk == 2:
                    total += 7
                else:
                    total += ni * mj * pk
    return total

def _build_decomposition(n_parts, m_parts, p_parts, n=5, m=5, p=5):
    """Build the factor matrices for given partitions."""
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
    
    total_rank = _compute_rank(n_parts, m_parts, p_parts)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    ni_blocks = len(n_parts)
    mj_blocks = len(m_parts)
    pk_blocks = len(p_parts)
    
    for i in range(ni_blocks):
        for j in range(mj_blocks):
            for k in range(pk_blocks):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen
                    i0, i1 = i_start, i_start + 1
                    j0, j1 = j_start, j_start + 1
                    k0, k1 = k_start, k_start + 1
                    
                    # M1 = (a00+a11)(b00+b11)
                    U[i0 * m + j0][col] = 1
                    U[i1 * m + j1][col] = 1
                    V[j0 * p + k0][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1
                    
                    # M2 = (a10+a11)b00
                    U[i1 * m + j0][col] = 1
                    U[i1 * m + j1][col] = 1
                    V[j0 * p + k0][col] = 1
                    W[k0 * n + i1][col] = 1
                    W[k1 * n + i1][col] = -1
                    col += 1
                    
                    # M3 = a00(b01-b11)
                    U[i0 * m + j0][col] = 1
                    V[j0 * p + k1][col] = 1
                    V[j1 * p + k1][col] = -1
                    W[k1 * n + i0][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1
                    
                    # M4 = a11(b10-b00)
                    U[i1 * m + j1][col] = 1
                    V[j1 * p + k0][col] = 1
                    V[j0 * p + k0][col] = -1
                    W[k0 * n + i0][col] = 1
                    W[k0 * n + i1][col] = 1
                    col += 1
                    
                    # M5 = (a00+a01)b11
                    U[i0 * m + j0][col] = 1
                    U[i0 * m + j1][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = -1
                    W[k1 * n + i0][col] = 1
                    col += 1
                    
                    # M6 = (a10-a00)(b00+b01)
                    U[i1 * m + j0][col] = 1
                    U[i0 * m + j0][col] = -1
                    V[j0 * p + k0][col] = 1
                    V[j0 * p + k1][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1
                    
                    # M7 = (a01-a11)(b10+b11)
                    U[i0 * m + j1][col] = 1
                    U[i1 * m + j1][col] = -1
                    V[j1 * p + k0][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = 1
                    col += 1
                    
                else:
                    # Schoolbook
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}

def construct():
    global _start_time, _TIME_LIMIT
    if time.time() - _start_time > _TIME_LIMIT:
        # Return a valid schoolbook decomposition as fallback
        n, m, p = DIMENSIONS
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
    
    n, m, p = DIMENSIONS
    
    # Generate all partitions of 5
    partitions = _get_partitions(5)
    
    # Find the best partition combination
    best_rank = float('inf')
    best_parts = None
    for n_parts in partitions:
        for m_parts in partitions:
            for p_parts in partitions:
                r = _compute_rank(n_parts, m_parts, p_parts)
                if r < best_rank:
                    best_rank = r
                    best_parts = (n_parts, m_parts, p_parts)
    
    # Build the decomposition with the best partitions
    result = _build_decomposition(best_parts[0], best_parts[1], best_parts[2], n, m, p)
    
    return result
