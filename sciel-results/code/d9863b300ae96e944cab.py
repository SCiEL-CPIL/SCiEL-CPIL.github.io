        for t in range(rank):
            for i in range(a_rows):
                for j in range(a_cols):
                    val = block_U[i*a_cols + j, t]
                    if val != 0:
                        # U shape (n*m, rank). Row index is flat index of A[i,j]
                        # A is n x m. Global row = A_row_offset + i. Global col = A_col_offset + j.
                        # Flat index = (A_row_offset + i) * m + (A_col_offset + j)
                        U_global[(A_row_offset + i) * m + (A_col_offset + j), next_col + t] += val
            for i in range(b_rows):
                for j in range(b_cols):
                    val = block_V[i*b_cols + j, t]
                    if val != 0:
                        # V shape (m*p, rank). Row index is flat index of B[i,j]
                        # B is m x p. Global row = B_row_offset + i. Global col = B_col_offset + j.
                        # Flat index = (B_row_offset + i) * p + (B_col_offset + j)
                        V_global[(B_row_offset + i) * p + (B_col_offset + j), next_col + t] += val
            for i in range(c_rows):
                for j in range(c_cols):
                    val = block_W[i*c_cols + j, t]
                    if val != 0:
                        # W shape (p*n, rank). Row index is k*n + i.
                        # C is n x p. Global row i_c = C_row_offset + i. Global col k_c = C_col_offset + j.
                        # W index = k_c * n + i_c
                        W_global[(C_col_offset + j) * n + (C_row_offset + i), next_col + t] += val
