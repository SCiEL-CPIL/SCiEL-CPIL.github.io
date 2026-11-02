# Inside the loop for R:
if R < n*m*p:
    # Warm start
    U_full = np.zeros((size_u, n*m*p))
    V_full = np.zeros((size_v, n*m*p))
    W_full = np.zeros((size_w, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_full[i*m+j, col] = 1
                V_full[j*p+k, col] = 1
                W_full[k*n+i, col] = 1
                col += 1
                
    # Select R random columns
    cols = np.random.choice(n*m*p, R, replace=False)
    params0 = np.concatenate([U_full[:, cols].flatten(), V_full[:, cols].flatten(), W_full[:, cols].flatten()])
else:
    params0 = np.random.randn((size_u + size_v + size_w) * R) * 0.1
