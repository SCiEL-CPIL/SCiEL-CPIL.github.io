    for i in range(16):
        c = i // 4
        a = i % 4
        w_row = (c + 4) * 4 + a
        W[w_row, 49:98] = W4[i, :]
