import time
_start_time = time.time()
TIME_LIMIT = 120.0

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x5 * 5x6 matrix multiplication.
    Uses block decomposition with Strassen's algorithm for 2x2x2 blocks.
    Target Rank: 84
    """
    n, m, p = 3, 5, 6
    
    # Partitions
    # n: [2, 1]
    # m: [2, 2, 1]
    # p: [2, 2, 2]
    
    n_parts = [(0, 2), (2, 3)]
    m_parts = [(0, 2), (2, 4), (4, 5)]
    p_parts = [(0, 2), (2, 4), (4, 6)]
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """
        Adds 7 Strassen terms for a 2x2x2 block.
        a_rows: list of 2 row indices for A
        c_cols: list of 2 col indices for A (row indices for B)
        d_cols: list of 2 col indices for B (row indices for C)
        """
        def U(la, lc):
            return a_rows[la] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[la]
            
        # M1 = (A00 + A11) * (B00 + B11)
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        # M2 = (A10 + A11) * B00
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        # M3 = A00 * (B01 - B11)
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        # M4 = A11 * (B10 - B00)
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        # M5 = (A00 + A01) * B11
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        # M6 = (A10 - A00) * (B00 + B01)
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        # M7 = (A01 - A11) * (B10 + B11)
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))

    def add_schoolbook(a_rows, c_cols, d_cols):
        """
        Adds schoolbook terms for a block.
        Rank = len(a_rows) * len(c_cols) * len(d_cols)
        """
        def U(la, lc):
            return a_rows[la] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[la]
            
        for la in range(len(a_rows)):
            for lc in range(len(c_cols)):
                for ld in range(len(d_cols)):
                    terms.append((
                        [(U(la, lc), 1)],
                        [(V(lc, ld), 1)],
                        [(W(ld, la), 1)]
                    ))

    # Iterate over blocks
    for ns, ne in n_parts:
        for ms, me in m_parts:
            for ps, pe in p_parts:
                if time.time() - _start_time > TIME_LIMIT - 5:
                    # Safety break, though construction is fast
                    break
                    
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                n_size = len(a_rows)
                m_size = len(c_cols)
                p_size = len(d_cols)
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                else:
                    add_schoolbook(a_rows, c_cols, d_cols)
    
    rank = len(terms)
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate matrices
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row][ci] += coeff
        for row, coeff in v_entries:
            V[row][ci] += coeff
        for row, coeff in w_entries:
            W[row][ci] += coeff
            
    return {"U": U, "V": V, "W": W}
