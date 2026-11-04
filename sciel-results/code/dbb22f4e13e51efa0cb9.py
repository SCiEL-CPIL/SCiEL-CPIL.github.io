    for r in range(8):
        # U part
        for i in range(2):
            U[i * m + 2][col] = 1
        # V part
        V[2 * p + r][col] = 1
        # W part
        W[r * n + 0][col] = 1
        W[r * n + 1][col] = 1
        col += 1
        if not _elapsed_ok():
            return _schoolbook()
