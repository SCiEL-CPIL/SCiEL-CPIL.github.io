    # Block 1 (cols 0..r_small-1)
    U[:, :r_small] = U_small
    for j in range(m_small):
        for k_prime in range(p_small):
            v_global = j * p + k_prime
            v_local = j * p_small + k_prime
            V[v_global, :r_small] = V_small[v_local, :]
            
    # Block 2 (cols r_small..2*r_small-1)
    U[:, r_small:] = U_small
    for j in range(m_small):
        for k_prime in range(p_small):
            v_global = j * p + (k_prime + p_small)
            v_local = j * p_small + k_prime
            V[v_global, r_small:] = V_small[v_local, :]
