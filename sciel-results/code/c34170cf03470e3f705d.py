def _verify_exact(U, V, W, n, m, p):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    un = int(round(u_val * 2))
                    vn = int(round(v_val * 2))
                    wn = int(round(w_val * 2))
                    total += un * vn * wn
                if total != 8:
                    return False
    return True
