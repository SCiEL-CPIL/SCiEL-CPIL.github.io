    def add_block_242(k_offset, r_offset):
        for r_A in range(7):
            for r_B in range(2):
                r = r_offset + r_A + 7 * r_B
                for i in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 + 2 * j2
                            # U
                            u_val = U_A[i * 2 + j1, r_A] * U_B[j2, r_B]
                            if u_val != 0:
                                U_total[i * 4 + j, r] += u_val
                            
                            for k_local in range(2):
                                k_global = k_offset + k_local
                                # V
                                v_val = V_A[j1 * 2 + k_local, r_A] * V_B[j2, r_B]
                                if v_val != 0:
                                    V_total[j * 7 + k_global, r] += v_val
                                # W
                                w_val = W_A[k_local * 2 + i, r_A] * W_B[0, r_B]
                                if w_val != 0:
                                    W_total[k_global * 2 + i, r] += w_val
