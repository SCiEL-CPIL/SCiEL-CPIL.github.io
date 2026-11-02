    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(4):
                for j1 in range(4):
                    u_val = U_s[i1][r1] * U_s[j1][r2]
                    if u_val != 0:
                        idx_u = i1 * 4 + j1
                        U_444[idx_u][r] += u_val
