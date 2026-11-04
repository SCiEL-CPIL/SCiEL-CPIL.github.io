def add_term(U, V, W, col, u_terms, v_terms, w_terms):
    for idx, coeff in u_terms:
        U[idx][col] += coeff
    for idx, coeff in v_terms:
        V[idx][col] += coeff
    for idx, coeff in w_terms:
        W[idx][col] += coeff
