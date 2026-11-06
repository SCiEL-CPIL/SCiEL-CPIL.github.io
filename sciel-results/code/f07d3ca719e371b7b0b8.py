def _compute_wire(S):
    if S in cache: return cache[S]
    # compute via balanced tree; but to allow sharing, recursively compute halves and cache them
    if len(S) == 1:
        return list(S)[0]
    # split S into two roughly equal halves
    lst = sorted(S)
    mid = len(lst)//2
    S1 = frozenset(lst[:mid])
    S2 = frozenset(lst[mid:])
    w1 = _compute_wire(S1)
    w2 = _compute_wire(S2)
    w = emit(w1, w2)
    cache[S] = w
    cache[S1] = w1
    cache[S2] = w2
    return w
