        def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
            nonlocal col_offset
            for idx, coeff in zip(u_indices, u_coeffs):
                U[idx, col_offset] += coeff
            for idx, coeff in zip(v_indices, v_coeffs):
                V[idx, col_offset] += coeff
            for idx, coeff in zip(w_indices, w_coeffs):
                W[idx, col_offset] += coeff
            col_offset += 1
