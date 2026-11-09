for j, c in enumerate(g_coeffs):
    if c:
        pos = (i + 12 - j) % n
        G_cyclic[i][pos] = 1
