    terms = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 0, 0),
        (2, 2, 2, 2)
    ]
    
    col = 0
    for j_off_a, j_off_b, k_off_b, k_off_c in terms:
        for a_coeffs, b_coeffs, c_coeffs in strassen_prods:
            for (i, j, val_a) in a_coeffs:
                u_idx = i * m + (j + j_off_a)
                U[u_idx][col] += val_a
            for (j, k, val_b) in b_coeffs:
                v_idx = (j + j_off_b) * p + (k + k_off_b)
                V[v_idx][col] += val_b
            for (i, k, val_c) in c_coeffs:
                w_idx = (k + k_off_c) * n + i
                W[w_idx][col] += val_c
            col += 1
