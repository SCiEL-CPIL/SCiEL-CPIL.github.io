    def pad_U(U_sub, n_sub, m_sub, n_total, m_total, i_offset, j_offset):
        # U_sub is n_sub*m_sub x rank
        # We need to place it in n_total*m_total x rank
        # Row index in global: (i_offset + i) * m_total + (j_offset + j)
        # This is not a simple block if m_total != m_sub.
        # If we split m, j_offset changes the column index in the row calculation.
        # Row = i * m_total + j.
        # If we split m into m1, m2.
        # Part 1: i in [0, n), j in [0, m1).
        # Part 2: i in [0, n), j in [m1, m1+m2).
        
        rank = U_sub.shape[1]
        U_full = np.zeros((n_total * m_total, rank))
        for i in range(n_sub):
            for j in range(m_sub):
                global_i = i_offset + i
                global_j = j_offset + j
                global_row = global_i * m_total + global_j
                local_row = i * m_sub + j
                U_full[global_row, :] = U_sub[local_row, :]
        return U_full
