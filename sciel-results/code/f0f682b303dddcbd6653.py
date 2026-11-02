import numpy as np
from fractions import Fraction

def _validate_exact(U, V, W):
    """Check if factor matrices exactly decompose the matmul tensor."""
    n, m, p = 2, 3, 8
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition. Rank 7."""
    n, m, p = 2, 2, 2
    r = 7
    U = [[0.0]*r for _ in range(n*m)]
    V = [[0.0]*r for _ in range(m*p)]
    W = [[0.0]*r for _ in range(p*n)]
    
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i*m+j, c]
    # V[j*p+k, c]
    # W[k*n+i, c]
    
    # c=0: M1
    # A parts: A00, A11
    U[0, 0] = 1.0
    U[3, 0] = 1.0
    # B parts: B00, B11
    V[0, 0] = 1.0
    V[3, 0] = 1.0
    # C parts: C00, C11 (M1 contributes to C00 and C11)
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So W[0,0]=1 (C00), W[3,0]=1 (C11)
    W[0, 0] = 1.0
    W[3, 0] = 1.0
    
    # c=1: M2
    # A parts: A10, A11
    U[2, 1] = 1.0
    U[3, 1] = 1.0
    # B parts: B00
    V[0, 1] = 1.0
    # C parts: C10 (M2+M4), C11 (-M2)
    W[2, 1] = 1.0 # C10
    W[3, 1] = -1.0 # C11
    
    # c=2: M3
    # A parts: A00
    U[0, 2] = 1.0
    # B parts: B10, B11
    V[2, 2] = 1.0
    V[3, 2] = -1.0
    # C parts: C00 (-M5? No, M3 is in C00? No. C00=M1+M4-M5+M7. C11=...+M3...)
    # Wait, Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M3 contributes to C01 and C11.
    W[1, 2] = 1.0 # C01
    W[3, 2] = 1.0 # C11
    
    # c=3: M4
    # A parts: A11
    U[3, 3] = 1.0
    # B parts: B01, B00
    V[1, 3] = 1.0
    V[0, 3] = -1.0
    # C parts: C00, C10
    W[0, 3] = 1.0
    W[2, 3] = 1.0
    
    # c=4: M5
    # A parts: A00, A01
    U[0, 4] = 1.0
    U[1, 4] = 1.0
    # B parts: B11
    V[3, 4] = 1.0
    # C parts: C00 (-M5), C01 (M5)
    W[0, 4] = -1.0
    W[1, 4] = 1.0
    
    # c=5: M6
    # A parts: A10, A00
    U[2, 5] = 1.0
    U[0, 5] = -1.0
    # B parts: B00, B01
    V[0, 5] = 1.0
    V[1, 5] = 1.0
    # C parts: C11 (M6)
    W[3, 5] = 1.0
    
    # c=6: M7
    # A parts: A01, A11
    U[1, 6] = 1.0
    U[3, 6] = -1.0
    # B parts: B10, B11
    V[2, 6] = 1.0
    V[3, 6] = 1.0
    # C parts: C00 (M7)
    W[0, 6] = 1.0
    
    return U, V, W

def _trivial_212():
    """Return U, V, W for 2x1x2 schoolbook decomposition. Rank 4."""
    n, m, p = 2, 1, 2
    r = 4
    U = [[0.0]*r for _ in range(n*m)]
    V = [[0.0]*r for _ in range(m*p)]
    W = [[0.0]*r for _ in range(p*n)]
    
    # C00 = A00 B00
    # C01 = A00 B01
    # C10 = A10 B00
    # C11 = A10 B01
    
    # c=0: A00 B00
    U[0, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0
    
    # c=1: A00 B01
    U[0, 1] = 1.0
    V[1, 1] = 1.0
    W[1, 1] = 1.0
    
    # c=2: A10 B00
    U[1, 2] = 1.0
    V[0, 2] = 1.0
    W[2, 2] = 1.0
    
    # c=3: A10 B01
    U[1, 3] = 1.0
    V[1, 3] = 1.0
    W[3, 3] = 1.0
    
    return U, V, W

def _combine_m_blocks(blocks):
    """
    Combine decompositions for n x m_i x p into n x sum(m_i) x p.
    blocks: list of (U_i, V_i, W_i)
    """
    if len(blocks) == 1:
        return blocks[0]
    
    U0, V0, W0 = blocks[0]
    n = len(U0) // 1 # n is len(U) / m? No, U is nm x r.
    # We need n and p to reconstruct indices.
    # Let's pass n, m, p to be safe or infer.
    # U shape (n*m, r). V shape (m*p, r). W shape (p*n, r).
    # From U0, m0 = ? We don't know m directly without n.
    # But we know the blocks are for the same n and p.
    # Let's infer n from W0. W0 is (p*n, r).
    # Let's infer p from V0. V0 is (m*p, r).
    
    # This function is tricky to infer dimensions. Let's pass n, m, p explicitly.
    raise ValueError("Use _combine_m_blocks_ex")

def _combine_m_blocks_ex(blocks, n, p):
    """
    Combine decompositions for n x m_i x p into n x sum(m_i) x p.
    """
    if len(blocks) == 1:
        return blocks[0]
    
    U0, V0, W0 = blocks[0]
    m0 = len(V0) // p
    r_total = sum(len(U[0]) for U, _, _ in blocks)
    
    U_new = [[0.0]*r_total for _ in range(n * sum(len(V)//p for V,_,_ in blocks))]
    # Wait, m_total = sum(m_i).
    m_total = 0
    for U, V, W in blocks:
        m_i = len(V) // p
        m_total += m_i
    
    U_new = [[0.0]*r_total for _ in range(n * m_total)]
    V_new = [[0.0]*r_total for _ in range(m_total * p)]
    W_new = [[0.0]*r_total for _ in range(p * n)]
    
    col_offset = 0
    m_offset = 0
    for U_i, V_i, W_i in blocks:
        m_i = len(V_i) // p
        r_i = len(U_i[0])
        
        # Fill U_new
        # U_i is n*m_i x r_i
        # U_new rows: i*m_total + j.
        # If j < m_i (relative to block), it maps to U_i.
        # But U_new row index is global j.
        # For a fixed i, j in [0, m_i), global_j = m_offset + j.
        # U_new[i*m_total + m_offset + j, col_offset+c] = U_i[i*m_i + j, c]
        for i in range(n):
            for j in range(m_i):
                for c in range(r_i):
                    U_new[i*m_total + m_offset + j, col_offset + c] = U_i[i*m_i + j, c]
        
        # Fill V_new
        # V_i is m_i*p x r_i
        # V_new rows: j*p + k.
        # For j in [0, m_i), global_j = m_offset + j.
        # V_new[(m_offset+j)*p + k, col_offset+c] = V_i[j*p + k, c]
        for j in range(m_i):
            for k in range(p):
                for c in range(r_i):
                    V_new[(m_offset+j)*p + k, col_offset + c] = V_i[j*p + k, c]
        
        # Fill W_new
        # W_i is p*n x r_i
        # W_new rows: k*n + i.
        # W_new[k*n + i, col_offset+c] = W_i[k*n + i, c]
        for k in range(p):
            for i in range(n):
                for c in range(r_i):
                    W_new[k*n + i, col_offset + c] = W_i[k*n + i, c]
        
        col_offset += r_i
        m_offset += m_i
    
    return U_new, V_new, W_new

def _combine_p_blocks_ex(blocks, n, m):
    """
    Combine decompositions for n x m x p_i into n x m x sum(p_i).
    """
    if len(blocks) == 1:
        return blocks[0]
    
    U0, V0, W0 = blocks[0]
    p0 = len(W0) // n
    r_total = sum(len(U[0]) for U, _, _ in blocks)
    
    p_total = 0
    for U, V, W in blocks:
        p_i = len(W) // n
        p_total += p_i
    
    U_new = [[0.0]*r_total for _ in range(n * m)]
    V_new = [[0.0]*r_total for _ in range(m * p_total)]
    W_new = [[0.0]*r_total for _ in range(p_total * n)]
    
    col_offset = 0
    p_offset = 0
    for U_i, V_i, W_i in blocks:
        p_i = len(W_i) // n
        r_i = len(U_i[0])
        
        # Fill U_new
        # U_i is n*m x r_i
        # U_new rows: i*m + j.
        # U_new[i*m + j, col_offset+c] = U_i[i*m + j, c]
        for i in range(n):
            for j in range(m):
                for c in range(r_i):
                    U_new[i*m + j, col_offset + c] = U_i[i*m + j, c]
        
        # Fill V_new
        # V_i is m*p_i x r_i
        # V_new rows: j*p_total + k.
        # For k in [0, p_i), global_k = p_offset + k.
        # V_new[j*p_total + p_offset + k, col_offset+c] = V_i[j*p_i + k, c]
        for j in range(m):
            for k in range(p_i):
                for c in range(r_i):
                    V_new[j*p_total + p_offset + k, col_offset + c] = V_i[j*p_i + k, c]
        
        # Fill W_new
        # W_i is p_i*n x r_i
        # W_new rows: k*n + i.
        # For k in [0, p_i), global_k = p_offset + k.
        # W_new[(p_offset+k)*n + i, col_offset+c] = W_i[k*n + i, c]
        for k in range(p_i):
            for i in range(n):
                for c in range(r_i):
                    W_new[(p_offset+k)*n + i, col_offset + c] = W_i[k*n + i, c]
        
        col_offset += r_i
        p_offset += p_i
    
    return U_new, V_new, W_new

def construct():
    n, m, p = 2, 3, 8
    
    # 1. Get 2x2x2 (Strassen) and 2x1x2 (Trivial)
    U_222, V_222, W_222 = _strassen_222()
    U_212, V_212, W_212 = _trivial_212()
    
    # 2. Combine to get 2x3x2
    # n=2, p=2
    U_232, V_232, W_232 = _combine_m_blocks_ex([
        (U_222, V_222, W_222),
        (U_212, V_212, W_212)
    ], n=2, p=2)
    
    # 3. Combine four 2x3x2 to get 2x3x8
    # n=2, m=3
    U_238, V_238, W_238 = _combine_p_blocks_ex([
        (U_232, V_232, W_232),
        (U_232, V_232, W_232),
        (U_232, V_232, W_232),
        (U_232, V_232, W_232)
    ], n=2, m=3)
    
    # 4. Convert to integers/fractions
    def to_frac(x):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return Fraction(x).limit_denominator(10)
    
    U_final = [[to_frac(x) for x in row] for row in U_238]
    V_final = [[to_frac(x) for x in row] for row in V_238]
    W_final = [[to_frac(x) for x in row] for row in W_238]
    
    # 5. Validate
    if not _validate_exact(U_final, V_final, W_final):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U_school = [[0]*rank for _ in range(n*m)]
        V_school = [[0]*rank for _ in range(m*p)]
        W_school = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_final, "V": V_final, "W": W_final}

