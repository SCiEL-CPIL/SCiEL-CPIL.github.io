for ci in range(n):
    for cj in range(m):
        for ck in range(p):
            u_idx = ci * m + cj
            v_idx = cj * p + ck
            w_idx = ck * n + ci
            prod_sum = float(np.sum(U_frac[:, u_idx] * V_frac[:, v_idx] * W_frac[:, w_idx]))
            if abs(prod_sum - 1.0) > 1e-9:
