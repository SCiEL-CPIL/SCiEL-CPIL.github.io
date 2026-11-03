            for i in range(n):
                for j in range(p):
                    for k in range(m):
                        C_computed[i, j] += U_np[i*m + j, :] * V_np[j*p + k, :] * W_np[k*n + i, :]
