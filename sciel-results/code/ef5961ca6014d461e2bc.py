    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal rank_offset
        
        # A is 2x2, B is 2x2, C is 2x2
        # Local indices 0,1
        
        # m1 = (A00 + A11) * (B00 + B11)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (0,1):1, (1,0):1, (1,1):-1  <-- C00+=m1, C01+=m1, C10+=m1, C11-=m1?
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Let's trace W contributions for each m.
        
        # m1:
        # U: {(0,0):1, (1,1):1}
        # V: {(0,0):1, (1,1):1}
        # W: {(0,0):1, (0,1):1, (1,0):1, (1,1):-1}
        # Check: C00 += 1*1*1 = 1. C01 += 1*1*1 = 1. C10 += 1*1*1 = 1. C11 += 1*1*(-1) = -1.
        # Correct for m1 terms in C.
        
        U[(a_row_start+0)*m + (a_col_start+0), rank_offset] = 1
        U[(a_row_start+1)*m + (a_col_start+1), rank_offset] = 1
        V[(b_row_start+0)*p + (b_col_start+0), rank_offset] = 1
        V[(b_row_start+1)*p + (b_col_start+1), rank_offset] = 1
        W[(c_col_start+0)*n + (c_row_start+0), rank_offset] = 1
        W[(c_col_start+1)*n + (c_row_start+0), rank_offset] = 1
        W[(c_col_start+0)*n + (c_row_start+1), rank_offset] = 1
        W[(c_col_start+1)*n + (c_row_start+1), rank_offset] = -1
        rank_offset += 1
        
        # m2 = (A10 + A11) * B00
        # U: {(1,0):1, (1,1):1}
        # V: {(0,0):1}
        # W: {(0,1):1}  # C10 += m2
        # C10 = m2 + m4. So C10 gets +1 from m2.
        
        U[(a_row_start+1)*m + (a_col_start+0), rank_offset] = 1
        U[(a_row_start+1)*m + (a_col_start+1), rank_offset] = 1
        V[(b_row_start+0)*p + (b_col_start+0), rank_offset] = 1
        W[(c_col_start+0)*n + (c_row_start+1), rank_offset] = 1
        rank_offset += 1
        
        # m3 = A00 * (B01 - B11)
        # U: {(0,0):1}
        # V: {(0,1):1, (1,1):-1}
        # W: {(0,1):1, (1,1):1} # C01 += m3, C11 += m3
        # C01 = m3 + m5. C11 = m1 - m2 + m3 + m6.
        
        U[(a_row_start+0)*m + (a_col_start+0), rank_offset] = 1
        V[(b_row_start+0)*p + (b_col_start+1), rank_offset] = 1
        V[(b_row_start+1)*p + (b_col_start+1), rank_offset] = -1
        W[(c_col_start+1)*n + (c_row_start+0), rank_offset] = 1
        W[(c_col_start+1)*n + (c_row_start+1), rank_offset] = 1
        rank_offset += 1
        
        # m4 = A11 * (B10 - B00)
        # U: {(1,1):1}
        # V: {(1,0):1, (0,0):-1}
        # W: {(0,0):1, (0,1):1} # C00 += m4, C10 += m4
        # C00 = m1 + m4 - m5 + m7. C10 = m2 + m4.
        
        U[(a_row_start+1)*m + (a_col_start+1), rank_offset] = 1
        V[(b_row_start+1)*p + (b_col_start+0), rank_offset] = 1
        V[(b_row_start+0)*p + (b_col_start+0), rank_offset] = -1
        W[(c_col_start+0)*n + (c_row_start+0), rank_offset] = 1
        W[(c_col_start+0)*n + (c_row_start+1), rank_offset] = 1
        rank_offset += 1
        
        # m5 = (A00 + A01) * B11
        # U: {(0,0):1, (0,1):1}
        # V: {(1,1):1}
        # W: {(0,1):1, (0,0):-1} # C01 += m5, C00 -= m5
        # C01 = m3 + m5. C00 = m1 + m4 - m5 + m7.
        
        U[(a_row_start+0)*m + (a_col_start+0), rank_offset] = 1
        U[(a_row_start+0)*m + (a_col_start+1), rank_offset] = 1
        V[(b_row_start+1)*p + (b_col_start+1), rank_offset] = 1
        W[(c_col_start+1)*n + (c_row_start+0), rank_offset] = 1
        W[(c_col_start+0)*n + (c_row_start+0), rank_offset] = -1
        rank_offset += 1
        
        # m6 = (A10 - A00) * (B00 + B01)
        # U: {(1,0):1, (0,0):-1}
        # V: {(0,0):1, (0,1):1}
        # W: {(1,1):1} # C11 += m6
        # C11 = m1 - m2 + m3 + m6.
        
        U[(a_row_start+1)*m + (a_col_start+0), rank_offset] = 1
        U[(a_row_start+0)*m + (a_col_start+0), rank_offset] = -1
        V[(b_row_start+0)*p + (b_col_start+0), rank_offset] = 1
        V[(b_row_start+0)*p + (b_col_start+1), rank_offset] = 1
        W[(c_col_start+1)*n + (c_row_start+1), rank_offset] = 1
        rank_offset += 1
        
        # m7 = (A01 - A11) * (B10 + B11)
        # U: {(0,1):1, (1,1):-1}
        # V: {(1,0):1, (1,1):1}
        # W: {(0,0):1} # C00 += m7
        # C00 = m1 + m4 - m5 + m7.
        
        U[(a_row_start+0)*m + (a_col_start+1), rank_offset] = 1
        U[(a_row_start+1)*m + (a_col_start+1), rank_offset] = -1
        V[(b_row_start+1)*p + (b_col_start+0), rank_offset] = 1
        V[(b_row_start+1)*p + (b_col_start+1), rank_offset] = 1
        W[(c_col_start+0)*n + (c_row_start+0), rank_offset] = 1
        rank_offset += 1
