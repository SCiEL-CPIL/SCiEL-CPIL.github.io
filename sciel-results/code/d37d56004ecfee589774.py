    g = [1]
    for r in roots:
        new_g = [0] * (len(g) + 1)
        for i in range(len(g)):
            # x * g_i x^i
            new_g[i+1] ^= g[i]
            # r * g_i x^i
            if g[i]:
                new_g[i] ^= gf_mul(g[i], r)
        g = new_g
