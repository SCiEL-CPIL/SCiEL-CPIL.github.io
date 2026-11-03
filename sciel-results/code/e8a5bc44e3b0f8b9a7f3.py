def _verify(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s += float(np.sum(u_row * v_row * w_row))
            if abs(s - 1.0) > tol:
                return False
    return True
