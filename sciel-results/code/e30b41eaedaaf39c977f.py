    covered_by = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        t = w
        d = [0] * 8
        for i in range(7, -1, -1):
            d[i] = t % 3
            t //= 3
        covered_by[w, 0] = w
        idx = 1
        for pos in range(8):
            weight = 3 ** (7 - pos)
            cur = d[pos]
            for nd in range(3):
                if nd == cur:
                    continue
                covered_by[w, idx] = w + (nd - cur) * weight
                idx += 1
