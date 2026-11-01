for p1, p2 in itertools.combinations(old_block, 2):
    if p1 > p2:
        p1, p2 = p2, p1
    # now p1 < p2
    ...
