"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17

# EVOLVE-BLOCK-START
def construct():
    comparators = []
    
    def bitonic_merge(n, start, ascending):
        if n <= 1:
            return
        k = n // 2
        for i in range(k):
            if ascending:
                comparators.append((start + i, start + k + i))
            else:
                comparators.append((start + k + i, start + i))
        bitonic_merge(k, start, ascending)
        bitonic_merge(k, start + k, ascending)
    
    def bitonic_sort(n, start, ascending=True):
        if n <= 1:
            return
        k = 1
        while k * 2 <= n:
            k *= 2
        # Sort first k ascending, next k descending
        bitonic_sort(k, start, True)
        if n - k > 1:
            bitonic_sort(n - k, start + k, False)
        # Now we have a bitonic sequence of length n
        # Merge it
        # The merge requires n/2 * log2(n) comparators
        _bitonic_merge_general(n, start, ascending)
    
    def _bitonic_merge_general(n, start, ascending):
        if n <= 1:
            return
        k = n // 2
        for i in range(k):
            if ascending:
                comparators.append((start + i, start + k + i))
            else:
                comparators.append((start + k + i, start + i))
        _bitonic_merge_general(k, start, ascending)
        if n - k > 1:
            _bitonic_merge_general(n - k, start + k, ascending)
    
    bitonic_sort(N, 0, True)
    return comparators
# EVOLVE-BLOCK-END
