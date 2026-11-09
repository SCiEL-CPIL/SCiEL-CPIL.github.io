    for lk in range(2):
        for li in range(2):
            W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
    