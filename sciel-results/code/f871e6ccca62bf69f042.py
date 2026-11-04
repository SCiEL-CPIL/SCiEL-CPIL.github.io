def b_block_indices(r_off, c_off):
    return [(r_off+a)*p + (c_off+b) for a in range(2) for b in range(2)]
