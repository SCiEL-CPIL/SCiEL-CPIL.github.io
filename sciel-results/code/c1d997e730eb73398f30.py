import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
        
    def add_strassen(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1
        add_term([(U(0,0), 1), (U(1,1), 1)],
                 [(V(0,0), 1), (V(1,1), 1)],
                 [(W(0,0), 1), (W(1,1), 1)])
        # M2
        add_term([(U(1,0), 1), (U(1,1), 1)],
                 [(V(0,0), 1)],
                 [(W(0,1), 1), (W(1,1), -1)])
        # M3
        add_term([(U(0,0), 1)],
                 [(V(0,1), 1), (V(1,1), -1)],
                 [(W(1,0), 1), (W(1,1), 1)])
        # M4
        add_term([(U(1,1), 1)],
                 [(V(1,0), 1), (V(0,0), -1)],
                 [(W(0,0), 1), (W(0,1), 1)])
        # M5
        add_term([(U(0,0), 1), (U(0,1), 1)],
                 [(V(1,1), 1)],
                 [(W(0,0), -1), (W(1,0), 1)])
        # M6
        add_term([(U(1,0), 1), (U(0,0), -1)],
                 [(V(0,0), 1), (V(0,1), 1)],
                 [(W(1,1), 1)])
        # M7
        add_term([(U(0,1), 1), (U(1,1), -1)],
                 [(V(1,0), 1), (V(1,1), 1)],
                 [(W(0,0), 1)])
                 
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0 = d_cols[0]
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[0]
        def W(ld, la): return d_cols[0]*n + a_rows[la]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)])
        add_term([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)])
        add_term([(U(1,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)])

    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U(la, lc): return a_rows[la]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)])
        add_term([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)])
        add_term([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)])

    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        def U(la, lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)])
        add_term([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)])

    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        def U(la, lc): return a_rows[0]*m + c_cols[0]
        def V(lc, ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[0]
        add_term([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)])
        add_term([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)])

    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row][ci] += coeff
        for row, coeff in v_entries:
            V[row][ci] += coeff
        for row, coeff in w_entries:
            W[row][ci] += coeff
            
    return {"U": U, "V": V, "W": W}
