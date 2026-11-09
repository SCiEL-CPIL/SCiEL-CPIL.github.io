def construct():
    # Generate the first 24 terms of the Golay-Rudin-Shapiro sequence
    # g(0) = 1
    # g(2n) = g(n)
    # g(2n+1) = -g(n)
    g = [1] * 24
    for n in range(1, 24):
        if n % 2 == 0:
            g[n] = g[n // 2]
        else:
            g[n] = -g[n // 2]
    return g
