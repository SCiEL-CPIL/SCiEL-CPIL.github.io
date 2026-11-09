def _distinct_degrees_tuple(combo):
    mask = 0
    for v in combo:
        mask |= (1 << v)
    degs = set()
    for v in combo:
        d = 0
        nb = _adj[v]
        for w in nb:
            if mask & (1 << w):
                d += 1
        degs.add(d)
    return len(degs)
