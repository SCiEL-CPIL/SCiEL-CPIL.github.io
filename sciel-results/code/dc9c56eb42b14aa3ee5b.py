def bitonic_sort(n, offset, direction):
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    
    res = []
    res += bitonic_sort(n // 2, offset, direction)
    res += bitonic_sort(n // 2, offset + n // 2, not direction)
    res += bitonic_merge(n // 2, offset, direction)
    return res

def bitonic_merge(n, offset, direction):
    if n <= 1:
        return [(offset, offset + 1)]
    
    res = []
    res += bitonic_merge(n // 2, offset, direction)
    res += bitonic_merge(n // 2, offset + n // 2, direction)
    for i in range(n // 2):
        res.append((offset + i, offset + n // 2 + i))
    return res
