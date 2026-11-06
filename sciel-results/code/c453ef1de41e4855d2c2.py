def _verify(net, bools):
    work = bools.copy()
    _simulate(net, work)
    return _is_sorted(work)
