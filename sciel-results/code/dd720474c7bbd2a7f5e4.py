for k, i, c in w_list:
    if c != 0:
        W[k * n + i][col] += c
