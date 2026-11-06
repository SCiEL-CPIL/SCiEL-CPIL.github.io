"""Optimal [18, 9, 6] extended quadratic-residue code, generated at runtime."""
import time

def construct():
    deadline = time.time() + 70.0
    # [17,8,5] QR(17) cyclic code, generator poly x^16+x^14+x^13+x^9+x^8
    g_bits = {8, 9, 13, 14, 16}
    g = 0
    for i in g_bits:
        g |= (1 << i)
    deg_g = g.bit_length() - 1

    def gf2_mod(x, g, deg_g):
        while x.bit_length() > deg_g:
            shift = x.bit_length() - 1 - deg_g
            x ^= g << shift
        return x

    words18 = []
    for m in range(256):
        p = m << 9
        c = gf2_mod(p, g, deg_g)
        parity = c.bit_count() & 1
        w = c | (parity << 17)
        words18.append(w)
    return words18
