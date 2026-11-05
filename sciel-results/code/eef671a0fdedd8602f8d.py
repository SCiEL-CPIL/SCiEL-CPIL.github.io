def calc_max_disc(signs, n=256, limit=None):
    max_disc = 1
    s = signs
    for d in range(1, n):
        max_len = (n + d - 1) // d
        if max_len <= max_disc:
            continue
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += s[idx]
                if curr < min_p: 
                    min_p = curr
                if curr > max_p: 
                    max_p = curr
                idx += d
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc
    return max_disc
