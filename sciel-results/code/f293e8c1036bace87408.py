    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle = list(range(N))
        elif L <= 20:
            cycle = list(range(L))
        else:
            d = N - L + 1
            cycle = [0] + list(range(d, N))
        cycles.append(cycle)
