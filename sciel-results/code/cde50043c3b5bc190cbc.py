for i in range(u_dim):
    i_a = i // m
    j_a = i % m
    for j in range(v_dim):
        j_a2 = j // p
        k_b = j % p
        if j_a != j_a2:
            continue
        for k in range(w_dim):
            k_b2 = k // n
            i_c = k % n
            if i_a != i_c or k_b != k_b2:
                continue
            T[i, j, k] = 1.0
