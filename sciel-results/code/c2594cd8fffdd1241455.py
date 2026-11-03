for kk in range(p):  # kk = k
    e = kk >> 1
    f = kk & 1
    for ii in range(n):  # ii = i
        a = ii >> 1
        b = ii & 1
        row = kk * n + ii  # W[k*n+i, r] ✓
