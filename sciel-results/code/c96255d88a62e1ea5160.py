for i in range(n):
    for j in range(m):
        for k in range(p):
            term = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
            C_computed[i, k] += A[i, j] * B[j, k] * term
