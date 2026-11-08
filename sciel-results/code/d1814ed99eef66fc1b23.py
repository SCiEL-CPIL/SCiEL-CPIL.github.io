    def add_schoolbook_2x2xK(a_base_i, b_base_i, c_base_i, k_base, k_len):
        for i in range(2):
            for j in range(2):
                for k in range(k_len):
                    U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                    U_term[u_idx(a_base_i + i, j)] = 1
                    V_term[v_idx(b_base_i + j, k_base + k)] = 1
                    W_term[w_idx(c_base_i + i, k_base + k)] = 1
                    add_rank1_term(U_term, V_term, W_term)
