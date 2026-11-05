    def min_dist_check(codes):
        n = len(codes)
        for i in range(n):
            if time.time() > deadline:
                return 100
            ci = codes[i]
            for j in range(i + 1, n):
                d = (ci ^ codes[j]).bit_count()
                if d < 6:
                    return d
        return 6
