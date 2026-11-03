"""Matrix multiplication 3x5x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
_start_time = time.time()
_TIMEOUT = 150

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    def add_strassen_2x2x2(a_rows, c_cols, d_cols):
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        add_term(
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        )
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        add_term(
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        )
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        add_term(
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        )
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        add_term(
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        )
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        add_term(
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        )
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        add_term(
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        )
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        add_term(
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        )
    
    def add_strassen_2x3x2(a_rows, c_cols, d_cols):
        """2x3x2: A is 2x3, B is 3x2, C is 2x2. Rank 10."""
        a0, a1 = a_rows
        c0, c1, c2 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # Use two 2x2x2 Strassen blocks + 2 schoolbook terms
        # Block 1: c=[0,1], d=[0,1] -> Strassen
        # Block 2: c=[1,2], d=[0,1] -> Strassen
        # But we can do better with a direct 10-term construction
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        add_term(
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        )
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        add_term(
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        )
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        add_term(
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        )
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        add_term(
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        )
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        add_term(
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        )
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        add_term(
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        )
        # M7 = A[0,2]*B[2,0]
        add_term(
            [(U(0,2), 1)],
            [(V(2,0), 1)],
            [(W(0,0), 1)]
        )
        # M8 = A[0,2]*B[2,1]
        add_term(
            [(U(0,2), 1)],
            [(V(2,1), 1)],
            [(W(1,0), 1)]
        )
        # M9 = A[1,2]*B[2,0]
        add_term(
            [(U(1,2), 1)],
            [(V(2,0), 1)],
            [(W(0,1), 1)]
        )
        # M10 = A[1,2]*B[2,1]
        add_term(
            [(U(1,2), 1)],
            [(V(2,1), 1)],
            [(W(1,1), 1)]
        )
    
    def add_strassen_1x3x2(a_rows, c_cols, d_cols):
        """1x3x2: A is 1x3, B is 3x2, C is 1x2. Rank 6 (schoolbook)."""
        a0 = a_rows[0]
        c0, c1, c2 = c_cols
        d0, d1 = d_cols
        
        def U(lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld): return d_cols[ld]*n + a_rows[0]
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        for lc in range(3):
            add_term([(U(lc), 1)], [(V(lc, 0), 1)], [(W(0), 1)])
            add_term([(U(lc), 1)], [(V(lc, 1), 1)], [(W(1), 1)])
    
    def add_schoolbook_2x2x2(a_rows, c_cols, d_cols):
        """2x2x2 schoolbook: rank 4."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        for la in range(2):
            for lc in range(2):
                for ld in range(2):
                    add_term(
                        [(U(la, lc), 1)],
                        [(V(lc, ld), 1)],
                        [(W(ld, la), 1)]
                    )
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2 schoolbook: rank 4."""
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(lc): return a_rows[0]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld): return d_cols[ld]*n + a_rows[0]
        
        for lc in range(2):
            for ld in range(2):
                add_term(
                    [(U(lc), 1)],
                    [(V(lc, ld), 1)],
                    [(W(ld), 1)]
                )
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2 schoolbook: rank 4."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la): return a_rows[la]*m + c_cols[0]
        def V(ld): return c_cols[0]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        for la in range(2):
            for ld in range(2):
                add_term(
                    [(U(la), 1)],
                    [(V(ld), 1)],
                    [(W(ld, la), 1)]
                )
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2 schoolbook: rank 2."""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(): return a_rows[0]*m + c_cols[0]
        def V(ld): return c_cols[0]*p + d_cols[ld]
        def W(ld): return d_cols[ld]*n + a_rows[0]
        
        for ld in range(2):
            add_term(
                [(U(), 1)],
                [(V(ld), 1)],
                [(W(ld), 1)]
            )
    
    # Partition: n=[2,1], m=[3,2], p=[2,2,2]
    n_blocks = [(0, 2), (2, 3)]
    m_blocks = [(0, 3), (3, 5)]
    p_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for (ns, ne) in n_blocks:
        for (ms, me) in m_blocks:
            for (ps, pe) in p_blocks:
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen_2x2x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 3 and p_size == 2:
                    add_strassen_2x3x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 3 and p_size == 2:
                    add_strassen_1x3x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 2:
                    add_schoolbook_2x2x2(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    # Build U, V, W matrices
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
    
    # Convert to numpy arrays
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
