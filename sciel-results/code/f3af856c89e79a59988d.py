    def add_term(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col
        for u_idx in u_indices:
            U[u_idx][col] += coeff
        for v_idx in v_indices:
            V[v_idx][col] += coeff
        for w_idx in w_indices:
            W[w_idx][col] += coeff
        col += 1
