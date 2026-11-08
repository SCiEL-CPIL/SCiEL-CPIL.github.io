import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_terms():
    terms = []
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}))
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}))
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}))
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}))
    return terms

def _strassen_4x4_block_terms():
    terms = []
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}))
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}))
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}))
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}))
    return terms

def _build_m444_decomp():
    """Build rank-49 decomposition of M_{4,4,4}.
    Returns list of (u_dict, v_dict, w_dict) where each dict maps (i,j) -> coeff.
    """
    s22 = _strassen_2x2_terms()
    s44 = _strassen_4x4_block_terms()
    terms = []
    for a_blocks, b_blocks, c_blocks in s44:
        for u2, v2, w2 in s22:
            u_full = {}
            v_full = {}
            w_full = {}
            for (bi, bj), cb in a_blocks.items():
                for (i, j), cu in u2.items():
                    key = (bi*2+i, bj*2+j)
                    u_full[key] = u_full.get(key, 0) + cu * cb
            for (bi, bj), cb in b_blocks.items():
                for (j, k), cv in v2.items():
                    key = (bi*2+j, bj*2+k)
                    v_full[key] = v_full.get(key, 0) + cv * cb
            for (bi, bj), cb in c_blocks.items():
                for (i, k), cw in w2.items():
                    key = (bi*2+i, bj*2+k)
                    w_full[key] = w_full.get(key, 0) + cw * cb
            terms.append((u_full, v_full, w_full))
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Build M_{4,4,4} decomposition (rank 49)
    m444_terms = _build_m444_decomp()
    
    # For M_{4,4,9}:
    # Columns 0-3: M_{4,4,4} part 1
    # Columns 4-7: M_{4,4,4} part 2
    # Column 8: schoolbook (rank 16)
    
    rank = 49 + 49 + 16
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank
    
    col = 0
    # Part 1: columns 0-3
    for u_dict, v_dict, w_dict in m444_terms:
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] += coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] += coeff
        col += 1
    
    # Part 2: columns 4-7 (offset k by 4)
    for u_dict, v_dict, w_dict in m444_terms:
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + (k + 4)][col] += coeff
        for (i, k), coeff in w_dict.items():
            W[(k + 4) * n + i][col] += coeff
        col += 1
    
    # Part 3: column 8, schoolbook
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
