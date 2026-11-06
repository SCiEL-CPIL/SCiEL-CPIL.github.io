    def v2(x):
        c = 0
        while x % 2 == 0:
            c += 1
            x //= 2
        return c
    by_v2_desc = sorted(moduli, key=lambda m: -v2(m))
    strategies = [None, asc, desc, by_deg_asc, by_deg_desc, by_v2_desc]
