    def add_strassen_2x2x2(a_rows, b_rows, b_cols):
        # a_rows: [r0, r1]
        # b_rows: [k0, k1]
        # b_cols: [c0, c1]
        
        # U indices
        u00 = a_rows[0] * m + 0 # Wait, A is n x m. Row i, Col j.
        # A indices: i in 0..n-1, j in 0..m-1.
        # U row index: i * m + j.
        # Here A1a is rows 0-1, cols 0-1.
        # So u00 = 0*4+0 = 0. u01 = 0*4+1 = 1.
        # u10 = 1*4+0 = 4. u11 = 1*4+1 = 5.
        
        # V indices: j * p + k.
        # B1a is rows 0-1, cols 0-1 (offset k_offset).
        # v00 = 0*6 + (k_offset+0) = k_offset.
        # v01 = 0*6 + (k_offset+1) = k_offset+1.
        # v10 = 1*6 + (k_offset+0) = 6+k_offset.
        # v11 = 1*6 + (k_offset+1) = 7+k_offset.
        
        # W indices: k * n + i.
        # C1a is rows 0-1, cols 0-1.
        # w00 = (k_offset+0)*3 + 0 = 3*k_offset.
        # w01 = (k_offset+0)*3 + 1 = 3*k_offset+1.
        # w10 = (k_offset+1)*3 + 0 = 3*k_offset+3.
        # w11 = (k_offset+1)*3 + 1 = 3*k_offset+4.
        
        u00, u01, u10, u11 = [a_rows[0]*m+0, a_rows[0]*m+1, a_rows[1]*m+0, a_rows[1]*m+1]
        v00, v01, v10, v11 = [b_rows[0]*p+b_cols[0], b_rows[0]*p+b_cols[1], b_rows[1]*p+b_cols[0], b_rows[1]*p+b_cols[1]]
        w00, w01, w10, w11 = [b_cols[0]*n+a_rows[0], b_cols[0]*n+a_rows[1], b_cols[1]*n+a_rows[0], b_cols[1]*n+a_rows[1]]
        
        # M1
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        # M2
        add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1)])
        # M3
        add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1)])
        # M4
        add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w10, 1), (w00, 1)])
        # M5
        add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w01, 1), (w11, -1)]) # Wait, C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # Let's re-verify Strassen C mapping.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1: W(w00, 1), W(w11, 1)
        # M2: W(w10, 1)
        # M3: W(w01, 1), W(w11, 1)
        # M4: W(w00, 1), W(w10, 1)
        # M5: W(w01, 1), W(w00, -1)
        # M6: W(w11, 1)
        # M7: W(w00, 1), W(w11, -1)
        
        # My add_term calls above were incomplete. I will rewrite the function properly.
