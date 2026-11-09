    poly = [1]
    for r in roots:
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            # new_poly[i] += c * r
            if c != 0:
                new_poly[i] ^= gf_mul(c, r)
            # new_poly[i+1] += c
            new_poly[i+1] ^= c
        poly = new_poly
