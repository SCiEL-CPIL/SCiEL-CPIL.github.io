    last_d = 0
    for d in range(1, max_N + 1):
        if time.time() > deadline - 20.0:
            break
        sigma[d::d] += d
        last_d = d
