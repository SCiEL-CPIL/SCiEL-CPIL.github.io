    for i in range(n):
        for k in range(p):
            s = 0
            for r in range(rank):
                u_val = 0
                for j in range(m):
                    u_val += U[i * m + j, r] * A[i, j]
                v_val = 0
                for j in range(m): # j corresponds to row of B
                    v_val += V[j * p + k, r] * B[j, k]
                s += W[k * n + i, r] * u_val * v_val
            C_computed[i, k] = s
