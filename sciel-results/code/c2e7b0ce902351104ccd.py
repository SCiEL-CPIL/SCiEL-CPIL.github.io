def make_perm(bit_order):
    # bit_order is a list of indices in the new order
    # e.g. [3, 1, 2, 0] means new_index = b3 b1 b2 b0
    n_bits = len(bit_order)
    size = 1 << n_bits
    perm = [0] * size
    for idx in range(size):
        # Extract bits
        bits = [(idx >> i) & 1 for i in range(n_bits)]
        # Reassemble
        new_idx = 0
        for new_pos, old_pos in enumerate(bit_order):
            if bits[old_pos]:
                new_idx |= (1 << new_pos)
        perm[idx] = new_idx
    return perm
