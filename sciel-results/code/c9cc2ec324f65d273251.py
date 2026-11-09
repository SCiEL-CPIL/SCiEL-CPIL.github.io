    idx_of_bit = {1 << i: i for i in range(N)}
    for mask in range(1, MAX_MASK):
        lsb = mask & -mask
        idx = idx_of_bit[lsb]
        prev_mask = mask ^ lsb
        next_a[mask] = next_a[prev_mask] | (1 << a[idx])
        next_b[mask] = next_b[prev_mask] | (1 << b[idx])
    