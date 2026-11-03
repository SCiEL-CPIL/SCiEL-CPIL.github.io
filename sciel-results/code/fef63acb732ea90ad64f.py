    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal term_idx
        if term_idx >= rank:
            return
        for idx, coeff in u_coeffs.items():
            if coeff != 0:
                U[idx][term_idx] += coeff
        for idx, coeff in v_coeffs.items():
            if coeff != 0:
                V[idx][term_idx] += coeff
        for idx, coeff in w_coeffs.items():
            if coeff != 0:
                W[idx][term_idx] += coeff
        term_idx += 1
    