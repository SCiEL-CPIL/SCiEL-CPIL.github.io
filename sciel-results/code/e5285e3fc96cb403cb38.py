    # Block 3: 4x1x5 (i=0..3, k=4, j=0..4)
    # Use rank 1 term
    u_vec = [(i * 5 + 4, 1) for i in range(4)]
    v_vec = [(4 * 5 + j, 1) for j in range(5)]
    w_vec = [(j * 4 + i, 1) for j in range(5) for i in range(4)]
    add_term(u_vec, v_vec, w_vec)
