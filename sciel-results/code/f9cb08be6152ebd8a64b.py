def bitonic_sort(n):
    # n must be power of 2
    if n == 1: return []
    comps = bitonic_sort(n//2) * 2  # sort two halves
    # bitonic merge
    comps += bitonic_merge(n)
    return comps

def bitonic_merge(n):
    # merge a bitonic sequence of length n
    if n == 1: return []
    # compare (i, i + n//2) for i in 0..n//2-1
    comps = [(i, i + n//2) for i in range(n//2)]
    # then recursively merge the two halves
    comps += bitonic_merge(n//2) * 2
    return comps
