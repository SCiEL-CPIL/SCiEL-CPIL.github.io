    pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
    # p is a tuple
    norm_pairs = tuple(tuple(sorted(p)) for p in pairs)
    