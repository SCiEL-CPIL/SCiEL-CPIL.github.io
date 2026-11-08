for i in range(n):
    for a in range(m):
        for k in range(p):
            acc = 0
            u_row = U[i * m + a]
            v_row = V[a * p + k]
            w_row = W[k * n + i]
            for c in range(rank):
                if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                    continue
                acc += u_row[c] * v_row[c] * w_row[c]
            if acc != 1:
                return False
