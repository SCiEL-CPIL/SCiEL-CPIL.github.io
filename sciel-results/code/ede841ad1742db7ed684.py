T = np.zeros((n, m, p))
for i in range(n):
    ...
    T[i, j, k] = np.dot(U[u_idx, :], V[v_idx, :])
