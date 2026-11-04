for i in range(n):
    for k in range(p):
        c_idx = k * n + i
        for j in range(m):
            a_idx = i * m + j
            b_idx = j * p + k
            s = 0
            for r in range(rank):
                if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                    s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
            if s != 1:
                return False
