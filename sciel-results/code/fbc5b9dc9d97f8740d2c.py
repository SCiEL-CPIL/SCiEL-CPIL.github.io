def add_term_signed(u_data, v_data, w_data):
    # u_data: list of (index, coeff)
    nonlocal col
    for idx, c in u_data:
        U[idx][col] += c
    for idx, c in v_data:
        V[idx][col] += c
    for idx, c in w_data:
        W[idx][col] += c
    col += 1
