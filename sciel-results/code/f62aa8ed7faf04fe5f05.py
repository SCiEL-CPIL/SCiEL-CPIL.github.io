while ai:
    lsb = ai & -ai
    j = lsb.bit_length() - 1
    if i < j:
        edges.append([i, j])
    ai ^= lsb
