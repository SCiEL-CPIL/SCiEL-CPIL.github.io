for x in range(1, 10*K+1):
    M = 0
    for k, v in zip(domain, values):
        if k <= x:
            M += v * (x // k)
