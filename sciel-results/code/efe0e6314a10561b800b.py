if g % 2 == 0:
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
