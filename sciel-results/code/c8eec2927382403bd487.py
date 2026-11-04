    def add_2x2_schoolbook(A_r, A_c, B_r, B_c, targets, col_offset):
        # targets: list of (C_r, C_c, sign)
        
        # Term 1: A00 * B00
        U[A_r*m + A_c][col_offset] = 1
        V[B_r*p + B_c][col_offset] = 1
        for C_r, C_c, s in targets:
            W[(C_c)*n + C_r][col_offset] += s
        col_offset += 1
        
        # Term 2: A01 * B01
        U[A_r*m + A_c + 1][col_offset] = 1
        V[B_r*p + B_c + 1][col_offset] = 1
        for C_r, C_c, s in targets:
            W[(C_c + 1)*n + C_r][col_offset] += s
        col_offset += 1
        
        # Term 3: A10 * B10
        U[A_r + 1*m + A_c][col_offset] = 1
        V[B_r + 1*p + B_c][col_offset] = 1
        for C_r, C_c, s in targets:
            W[(C_c)*n + C_r + 1][col_offset] += s
        col_offset += 1
        
        # Term 4: A11 * B11
        U[A_r + 1*m + A_c + 1][col_offset] = 1
        V[B_r + 1*p + B_c + 1][col_offset] = 1
        for C_r, C_c, s in targets:
            W[(C_c + 1)*n + C_r + 1][col_offset] += s
        col_offset += 1
        
        return col_offset
