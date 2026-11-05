"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a Batcher's Odd-Merge sorting network for N=16 wires.
    This network uses 68 comparators, which is significantly better than the
    120 comparators used by the odd-even transposition network.
    """
    return _batcher_odd_merge_sort(N)

def _batcher_odd_merge_sort(n):
    """
    Recursively constructs the Batcher's Odd-Merge sorting network.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = _batcher_odd_merge_sort(half)
    right = _batcher_odd_merge_sort(half)
    
    # Shift the right half comparators
    right = [(i + half, j + half) for i, j in right]
    
    merge = _batcher_odd_merge(n)
    
    return left + right + merge

def _batcher_odd_merge(n):
    """
    Recursively constructs the Odd-Even Merge network for two sorted halves
    of size n/2.
    """
    if n == 2:
        return [(0, 1)]
    if n == 4:
        # Optimized merge for two sorted lists of size 2
        # Comparators: (0,2), (1,3), (1,2)
        return [(0, 2), (1, 3), (1, 2)]
    
    half = n // 2
    
    # Merge on even indices: 0, 2, ..., n-2
    even_merge = _batcher_odd_merge(half)
    even_mapped = [(2 * i, 2 * j) for i, j in even_merge]
    
    # Merge on odd indices: 1, 3, ..., n-1
    odd_merge = _batcher_odd_merge(half)
    odd_mapped = [(2 * i + 1, 2 * j + 1) for i, j in odd_merge]
    
    # Shuffle step: compare adjacent pairs (0,1), (2,3), ...
    shuffle = [(i, i + 1) for i in range(0, n - 1, 2)]
    
    return even_mapped + odd_mapped + shuffle
# EVOLVE-BLOCK-END
