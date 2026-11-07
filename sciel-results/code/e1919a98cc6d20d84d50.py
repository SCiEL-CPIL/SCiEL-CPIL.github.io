def construct():
    start = time.time()
    candidates = []
    for t in range(5):
        for s in range(2):
            sq = _build_square(t, s)
            if _valid_lsq(sq):
                candidates.append(sq)
    
    best_set = []
    for sq in candidates:
        ok = True
        for other in best_set:
            if not _orthogonal(sq, other):
                ok = False
                break
        if ok:
            best_set.append(sq)
            if len(best_set) >= 2:
                break
    return best_set
