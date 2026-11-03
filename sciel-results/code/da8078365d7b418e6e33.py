for bi in range(2):  # block row of A / block col of C
    for bj in range(2):  # block col of A / block row of B
        for bk in range(2):  # block col of B / block row of C
            for (i, j, k, a, b, c) in terms_3x3:
