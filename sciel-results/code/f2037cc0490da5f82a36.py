    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    u_val = U[i*m+j][c]
                    v_val = V[j*p+k][c]
                    w_val = W[k*n+i][c]
                    s += u_val * v_val * w_val
                if abs(s - 1.0) > 1e-8:
                    return False
    