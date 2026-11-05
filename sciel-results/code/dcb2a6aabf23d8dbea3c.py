<<<<<<< SEARCH
def _build_block_decomp(n, m, p, n_splits, m_splits, p_splits):
    """Build full decomposition from block splits."""
    # Compute total rank
    total_rank = 0
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                total_rank += _block_rank(ns, ms, ps)

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Compute offsets
    n_offsets = []
    off = 0
    for ns in n_splits:
        n_offsets.append(off)
        off += ns
    m_offsets = []
    off = 0
    for ms in m_splits:
        m_offsets.append(off)
        off += ms
    p_offsets = []
    off = 0
    for ps in p_splits:
        p_offsets.append(off)
        off += ps

    col_offset = 0
    for ni, ns in enumerate(n_splits):
        for mi, ms in enumerate(m_splits):
            for pi, ps in enumerate(p_splits):
                U_b, V_b, W_b = _get_block_decomp(ns, ms, ps)
                r_b = len(U_b[0])
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]

                for i in range(ns):
                    for j in range(ms):
                        for c in range(r_b):
                            U[(n_off + i) * m + (m_off + j)][col_offset + c] = U_b[i * ms + j][c]

                for j in range(ms):
                    for k in range(ps):
                        for c in range(r_b):
                            V[(m_off + j) * p + (p_off + k)][col_offset + c] = V_b[j * ps + k][c]

                for k in range(ps):
                    for i in range(ns):
                        for c in range(r_b):
                            W[(p_off + k) * n + (n_off + i)][col_offset + c] = W_b[k * ns + i][c]

                col_offset += r_b

    return U, V, W, total_rank
=======
def _build_block_decomp(n, m, p, n_splits, m_splits, p_splits):
    """Build full decomposition from block splits."""
    # Compute total rank
    total_rank = 0
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                total_rank += _block_rank(ns, ms, ps)

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Compute offsets
    n_offsets = []
    off = 0
    for ns in n_splits:
        n_offsets.append(off)
        off += ns
    m_offsets = []
    off = 0
    for ms in m_splits:
        m_offsets.append(off)
        off += ms
    p_offsets = []
    off = 0
    for ps in p_splits:
        p_offsets.append(off)
        off += ps

    col_offset = 0
    for ni, ns in enumerate(n_splits):
        for mi, ms in enumerate(m_splits):
            for pi, ps in enumerate(p_splits):
                U_b, V_b, W_b = _get_block_decomp(ns, ms, ps)
                r_b = len(U_b[0])
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]

                for i in range(ns):
                    for j in range(ms):
                        src_row = i * ms + j
                        dst_row = (n_off + i) * m + (m_off + j)
                        for c in range(r_b):
                            val = U_b[src_row][c]
                            if val != 0:
                                U[dst_row][col_offset + c] = val

                for j in range(ms):
                    for k in range(ps):
                        src_row = j * ps + k
                        dst_row = (m_off + j) * p + (p_off + k)
                        for c in range(r_b):
                            val = V_b[src_row][c]
                            if val != 0:
                                V[dst_row][col_offset + c] = val

                for k in range(ps):
                    for i in range(ns):
                        src_row = k * ns + i
                        dst_row = (p_off + k) * n + (n_off + i)
                        for c in range(r_b):
                            val = W_b[src_row][c]
                            if val != 0:
                                W[dst_row][col_offset + c] = val

                col_offset += r_b

    return U, V, W, total_rank
>>>>>>> REPLACE

