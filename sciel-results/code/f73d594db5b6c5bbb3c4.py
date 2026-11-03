    def add_strassen_block(u_offset_j, v_offset_l, col_start):
        # u_offset_j: offset for j (col of A, row of B)
        # v_offset_l: offset for l (col of B, col of C)
        # Note: v_offset_k is NOT used because B's row index is the SAME as A's col index j.
        # Wait, B is 4x6. A is 2x4.
        # A col j is 0..3. B row j is 0..3.
        # In Strassen 2x2x2, A is 2x2, B is 2x2.
        # A col j_local 0..1. B row j_local 0..1.
        # So they share the same local index j_local.
        # Global J = j_local + u_offset_j.
        # So V row index is (j_local + u_offset_j) * p + (l_local + v_offset_l).
        
        # M1: (A00+A11)(B00+B11)
        # A00: i=0, j=0. A11: i=1, j=1.
        # B00: j=0, l=0. B11: j=1, l=1.
        U[0 * m + (0 + u_offset_j)][col_start] = 1
        U[1 * m + (1 + u_offset_j)][col_start] = 1
        V[(0 + u_offset_j) * p + (0 + v_offset_l)][col_start] = 1
        V[(1 + u_offset_j) * p + (1 + v_offset_l)][col_start] = 1
        W[(0 + v_offset_l) * n + 0][col_start] += 1
        W[(1 + v_offset_l) * n + 1][col_start] += 1
