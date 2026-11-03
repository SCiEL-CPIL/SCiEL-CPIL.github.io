    def embed(n, m, p, sub_n, sub_m, sub_p, off_i, off_j, off_k, d_sub):
        U_sub, V_sub, W_sub, r = d_sub
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for i in range(sub_n):
            for j in range(sub_m):
                for k in range(sub_p):
                    u_idx = (off_i + i) * m + (off_j + j)
                    v_idx = (off_j + j) * p + (off_k + k)
                    w_idx = (off_k + k) * n + (off_i + i)
                    
                    U[u_idx, :] = U_sub[i * sub_m + j, :]
                    V[v_idx, :] = V_sub[j * sub_p + k, :]
                    W[w_idx, :] = W_sub[k * sub_n + i, :]
        return U, V, W, r
