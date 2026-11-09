N = 16

def _batcher_merge(n, o1, o2):
    if n == 1:
        return [(o1, o2)]
    h = n // 2
    comps = []
    comps += _batcher_merge(h, o1, o2)
    comps += _batcher_merge(h, o1 + h, o2 + h)
    for i in range(0, n, 2):
        comps.append((o1 + i, o1 + i + 1))
        comps.append((o2 + i, o2 + i + 1))
    return comps

def _batcher_sort(n):
    if n == 1:
        return []
    left = _batcher_sort(n // 2)
    right = _batcher_sort(n // 2)
    right = [(i + n // 2, j + n // 2) for i, j in right]
    merge = _batcher_merge(n // 2, 0, n // 2)
    return left + right + merge

def construct():
    return _batcher_sort(N)
