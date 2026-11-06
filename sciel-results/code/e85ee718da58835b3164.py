x = nu
while x:
    low = x & (-x)
    xi = low.bit_length() - 1
    if adj_bitmask[xi] & nv:
        return True
    x ^= low
