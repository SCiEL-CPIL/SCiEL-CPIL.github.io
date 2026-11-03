"""Strassen 2x2 blocks for 3x5x7 matrix multiplication, rank 99."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    m_offsets = []
    off = 0
    for s in m_splits:
        m_offsets.append(off)
        off += s
    p_offsets = []
    off = 0
    for s in p_splits:
        p_offsets.append(off)
        off += s
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """u_coeff, v_coeff, w_coeff are dicts: {(row): val}"""
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        for (r, c) in u_coeff.items():
            U_row[r] = c
        for (r, c) in v_coeff.items():
            V_row[r] = c
        for (r, c) in w_coeff.items():
            W_row[r] = c
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    def A_idx(i, j):
        return i * m + j
    
    def B_idx(j, k):
        return j * p + k
    
    def C_idx(i, k):
        return k * n + i
    
    for mi_idx, mi in enumerate(m_splits):
        m_off = m_offsets[mi_idx]
        for pi_idx, pi in enumerate(p_splits):
            p_off = p_offsets[pi_idx]
            
            if mi == 2 and pi == 2:
                # Strassen on (3,2)x(2,2)->(3,2)
                # A_top: rows 0,1; cols m_off, m_off+1
                # A_bot: row 2; cols m_off, m_off+1
                # B: rows m_off, m_off+1; cols p_off, p_off+1
                a11, a12 = A_idx(0, m_off), A_idx(0, m_off + 1)
                a21, a22 = A_idx(1, m_off), A_idx(1, m_off + 1)
                b11, b12 = B_idx(m_off, p_off), B_idx(m_off, p_off + 1)
                b21, b22 = B_idx(m_off + 1, p_off), B_idx(m_off + 1, p_off + 1)
                c11, c12 = C_idx(0, p_off), C_idx(0, p_off + 1)
                c21, c22 = C_idx(1, p_off), C_idx(1, p_off + 1)
                
                # M1 = (a11+a22)(b11+b22) -> C11 += M1, C22 += M1
                add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
                # M2 = (a21+a22)b11 -> C21 += M2, C22 -= M2
                add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
                # M3 = a11(b21-b22) -> C12 += M3, C22 += M3
                add_term({a11: 1}, {b21: 1, b22: -1}, {c12: 1, c22: 1})
                # M4 = a22(b11-b21) -> C11 += M4, C21 += M4
                add_term({a22: 1}, {b11: 1, b21: -1}, {c11: 1, c21: 1})
                # M5 = (a11+a12)b22 -> C11 -= M5, C12 += M5
                add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
                # M6 = (a21-a11)(b11+b12) -> C22 += M6
                add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
                # M7 = (a12-a22)(b21+b22) -> C11 += M7
                add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})
                
                # A_bot * B: row 2 of A times B
                a31, a32 = A_idx(2, m_off), A_idx(2, m_off + 1)
                c31, c32 = C_idx(2, p_off), C_idx(2, p_off + 1)
                # C31 = a31*b11 + a32*b21
                add_term({a31: 1}, {b11: 1}, {c31: 1})
                add_term({a32: 1}, {b21: 1}, {c31: 1})
                # C32 = a31*b12 + a32*b22
                add_term({a31: 1}, {b12: 1}, {c32: 1})
                add_term({a32: 1}, {b22: 1}, {c32: 1})
            else:
                for i in range(n):
                    for j in range(mi):
                        for k in range(pi):
                            add_term({A_idx(i, m_off + j): 1},
                                     {B_idx(m_off + j, p_off + k): 1},
                                     {C_idx(i, p_off + k): 1})
    
    rank = len(U_list)
    U = U_list
    V = V_list
    W = W_list
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
