def fill(S, counts):
    for _ in range(20):
        comp = [x for x in range(1,1001) if x not in S]
        rng.shuffle(comp)
        added = False
        for x in comp:
            if _valid_add(x, S, counts):
                _do_add(x, S, counts)
                added = True
        if not added:
            break
