"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _verify_network_vectorized(comparators):
    """Verify sorting network using zero-one principle, vectorized over all 2^16 inputs."""
    masks = np.arange(1 << N, dtype=np.uint32)
    bits = np.empty((1 << N, N), dtype=np.int8)
    for i in range(N):
        bits[:, i] = ((masks >> i) & 1).astype(np.int8)
    
    wires = bits.copy()
    for i, j in comparators:
        a = wires[:, i]
        b = wires[:, j]
        wires[:, i] = np.minimum(a, b)
        wires[:, j] = np.maximum(a, b)
    
    # Check sorted
    for i in range(N - 1):
        if not np.all(wires[:, i] <= wires[:, i + 1]):
            return False
    return True

def _bitonic_sort_network(n):
    """Generate bitonic sort network."""
    if n == 1:
        return []
    comparators = []
    
    def merge(start, length, ascending):
        if length == 1:
            return
        half = length // 2
        for i in range(half):
            l, r = start + i, start + i + half
            if ascending:
                comparators.append((l, r))
            else:
                comparators.append((r, l))
        merge(start, half, ascending)
        merge(start + half, half, ascending)
    
    def sort_rec(start, length, ascending):
        if length == 1:
            return
        half = length // 2
        sort_rec(start, half, True)
        sort_rec(start + half, half, False)
        merge(start, length, ascending)
    
    sort_rec(0, n, True)
    return comparators

def construct():
    """Return a sorting network for 16 wires minimizing comparator count."""
    deadline = time.time() + 80
    
    # Start with bitonic sort
    best = _bitonic_sort_network(N)
    
    # Local search: iteratively try removing comparators
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for idx in range(len(best)):
            if time.time() > deadline - 3:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_vectorized(candidate):
                best = candidate
                improved = True
                break
    
    return best
# EVOLVE-BLOCK-END

