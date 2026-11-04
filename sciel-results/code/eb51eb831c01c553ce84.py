"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 110.0

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Partition indices
    # n (rows of A): [0,1] | [2]
    # m (cols of A / rows of B): [0,1] | [2,3] | [4]
    # p (cols of B): [0,1] | [2,3] | [4,5]
    
    n_parts = [[0, 1], [2]]
    m_parts = [[0, 1], [2, 3], [4]]
    p_parts = [[0, 1], [2, 3], [4, 5]]
    
    terms = []  # each term: (list of (u_row, coeff), list of (v_row, coeff), list of (w_row, coeff))
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    def add_strassen_2x2x2(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        add_term(
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        )
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += 1, C[1,1] += -1
        add_term(
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        )
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        add_term(
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        )
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        add_term(
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        )
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] += -1, C[0,1] += 1
        add_term(
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        )
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += 1
        add_term(
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        )
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += 1
        add_term(
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        )
    
    def add_2x1x2_rank3(a_rows, c_cols, d_cols):
        """2x1x2 block with rank 3: A is 2x1, B is 1x2, C is 2x2."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la): return a_rows[la]*m + c0
        def V(ld): return c0*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # C[0,0] = A[0]*B[0,0]
        # C[0,1] = A[0]*B[0,1]
        # C[1,0] = A[1]*B[0,0]
        # C[1,1] = A[1]*B[0,1]
        
        # M1 = A[0] * B[0,0] -> C[0,0]
        add_term([(U(0), 1)], [(V(0), 1)], [(W(0,0), 1)])
        
        # M2 = A[1] * B[0,1] -> C[1,1]
        add_term([(U(1), 1)], [(V(1), 1)], [(W(1,1), 1)])
        
        # M3 = (A[0]+A[1]) * (B[0,0]+B[0,1]) -> C[0,0] += 1, C[0,1] += 1, C[1,0] += 1, C[1,1] += 1
        # But we already have C[0,0] and C[1,1], so we need to subtract
        # Actually let's use: M3 = (A[0]+A[1]) * (B[0,0]+B[0,1])
        # C[0,0] = M1 + M3 - M2... no
        # Let me think: C[0,0] = A0*B00, C[0,1] = A0*B01, C[1,0] = A1*B00, C[1,1] = A1*B01
        # M1 = A0*B00
        # M2 = A1*B01
        # M3 = (A0+A1)*(B00+B01) = A0*B00 + A0*B01 + A1*B00 + A1*B01
        # C[0,0] = M1
        # C[0,1] = M3 - M1 - M2
        # C[1,0] = M3 - M1 - M2... no that's not right
        # C[0,1] = A0*B01 = M3 - M1 - M2 - A1*B00... 
        # This doesn't work with just 3 terms in a simple way.
        
        # Alternative: use rank 4 (schoolbook) for 2x1x2
        # Actually, the rank of 2x1x2 is 4 (it's a 4-dimensional tensor space)
        # Let me just use schoolbook for these.
        pass
    
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: schoolbook rank 4."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(la): return a_rows[la]*m + c0
        def V(ld): return c0*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        add_term([(U(0), 1)], [(V(0), 1)], [(W(0,0), 1)])
        add_term([(U(0), 1)], [(V(1), 1)], [(W(1,0), 1)])
        add_term([(U(1), 1)], [(V(0), 1)], [(W(0,1), 1)])
        add_term([(U(1), 1)], [(V(1), 1)], [(W(1,1), 1)])
    
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: schoolbook rank 4."""
        a0 = a_rows[0]
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(lc): return a0*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld): return d_cols[ld]*n + a0
        
        add_term([(U(0), 1)], [(V(0,0), 1)], [(W(0), 1)])
        add_term([(U(0), 1)], [(V(0,1), 1)], [(W(1), 1)])
        add_term([(U(1), 1)], [(V(1,0), 1)], [(W(0), 1)])
        add_term([(U(1), 1)], [(V(1,1), 1)], [(W(1), 1)])
    
    def add_schoolbook_2x1x1(a_rows, c_cols, d_cols):
        """2x1x1: schoolbook rank 2."""
        a0, a1 = a_rows
        c0 = c_cols[0]
        d0 = d_cols[0]
        
        def U(la): return a_rows[la]*m + c0
        def V(): return c0*p + d0
        def W(la): return d0*n + a_rows[la]
        
        add_term([(U(0), 1)], [(V(), 1)], [(W(0), 1)])
        add_term([(U(1), 1)], [(V(), 1)], [(W(1), 1)])
    
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: schoolbook rank 2."""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0, d1 = d_cols
        
        def U(): return a0*m + c0
        def V(ld): return c0*p + d_cols[ld]
        def W(ld): return d_cols[ld]*n + a0
        
        add_term([(U(), 1)], [(V(0), 1)], [(W(0), 1)])
        add_term([(U(), 1)], [(V(1), 1)], [(W(1), 1)])
    
    def add_schoolbook_1x1x1(a_rows, c_cols, d_cols):
        """1x1x1: rank 1."""
        a0 = a_rows[0]
        c0 = c_cols[0]
        d0 = d_cols[0]
        
        def U(): return a0*m + c0
        def V(): return c0*p + d0
        def W(): return d0*n + a0
        
        add_term([(U(), 1)], [(V(), 1)], [(W(), 1)])
    
    # Now iterate through all block combinations
    for ni, a_rows in enumerate(n_parts):
        for mi, c_cols in enumerate(m_parts):
            for pi, d_cols in enumerate(p_parts):
                na, ma, pa = len(a_rows), len(c_cols), len(d_cols)
                
                if na == 2 and ma == 2 and pa == 2:
                    add_strassen_2x2x2(a_rows, c_cols, d_cols)
                elif na == 2 and ma == 1 and pa == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif na == 1 and ma == 2 and pa == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif na == 2 and ma == 1 and pa == 1:
                    add_schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif na == 1 and ma == 1 and pa == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif na == 1 and ma == 1 and pa == 1:
                    add_schoolbook_1x1x1(a_rows, c_cols, d_cols)
                elif na == 2 and ma == 2 and pa == 1:
                    # 2x2x1: use schoolbook rank 4
                    a0, a1 = a_rows
                    c0, c1 = c_cols
                    d0 = d_cols[0]
                    def U(la, lc): return a_rows[la]*m + c_cols[lc]
                    def V(lc): return c_cols[lc]*p + d0
                    def W(la): return d0*n + a_rows[la]
                    add_term([(U(0,0), 1)], [(V(0), 1)], [(W(0), 1)])
                    add_term([(U(0,1), 1)], [(V(1), 1)], [(W(0), 1)])
                    add_term([(U(1,0), 1)], [(V(0), 1)], [(W(1), 1)])
                    add_term([(U(1,1), 1)], [(V(1), 1)], [(W(1), 1)])
                elif na == 1 and ma == 2 and pa == 1:
                    # 1x2x1: schoolbook rank 2
                    a0 = a_rows[0]
                    c0, c1 = c_cols
                    d0 = d_cols[0]
                    def U(lc): return a0*m + c_cols[lc]
                    def V(lc): return c_cols[lc]*p + d0
                    def W(): return d0*n + a0
                    add_term([(U(0), 1)], [(V(0), 1)], [(W(), 1)])
                    add_term([(U(1), 1)], [(V(1), 1)], [(W(), 1)])
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row][col] += coeff
        for row, coeff in v_entries:
            V[row][col] += coeff
        for row, coeff in w_entries:
            W[row][col] += coeff
    
    # Verify
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = int(np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :]))
                if val != 1:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 0:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col_s] = 1
                    V_s[j*p+k][col_s] = 1
                    W_s[k*n+i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
