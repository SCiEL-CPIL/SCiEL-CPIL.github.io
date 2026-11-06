"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

# Deadline guard: evaluator kills after 90s; use 85s as safety margin
_DEADLINE = time.time() + 85

def _time_remaining():
    return _DEADLINE - time.time()

def _verify_network(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle."""
    for bits in itertools.product([0, 1], repeat=n):
        vals = list(bits)
        for i, j in network:
            if vals[i] > vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
        if vals != sorted(vals):
            return False
    return True

def _verify_network_fast(network, n=16):
    """Faster verification using bitmasks."""
    for mask in range(1 << n):
        vals = [(mask >> i) & 1 for i in range(n)]
        for i, j in network:
            if vals[i] > vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
        for i in range(n - 1):
            if vals[i] > vals[i + 1]:
                return False
    return True

def _batcher_odd_even_mergesort(n):
    """Generate Batcher's odd-even mergesort network for n wires.
    Returns list of (i, j) comparator pairs."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 4:
        return [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
    
    half = n // 2
    # Sort left half
    left = _batcher_odd_even_mergesort(half)
    # Sort right half (offset by half)
    right = [(i + half, j + half) for i, j in _batcher_odd_even_mergesort(half)]
    
    # Merge the two sorted halves using Batcher's odd-even merge
    # 1. Compare corresponding elements: (i, i+half) for i in range(half)
    merge = [(i, i + half) for i in range(half)]
    
    # 2. Recursively sort the even-indexed subsequence (wires 0, 2, 4, ...)
    #    and odd-indexed subsequence (wires 1, 3, 5, ...)
    even_subseq = _batcher_odd_even_mergesort(half)
    even_mapped = [(2 * a, 2 * b) for a, b in even_subseq]
    
    odd_subseq = _batcher_odd_even_mergesort(half)
    odd_mapped = [(2 * a + 1, 2 * b + 1) for a, b in odd_subseq]
    
    return left + right + merge + even_mapped + odd_mapped

def _remove_redundant_comparators(network, n=16):
    """Try to remove comparators that don't affect correctness."""
    if not network:
        return network
    
    # Build a set of all comparators for quick lookup
    comp_set = set(network)
    
    # Try removing each comparator and checking if the network still sorts
    # This is expensive, so we'll use a heuristic: try removing in reverse order
    # (later comparators are more likely to be redundant if earlier ones suffice)
    
    # First, let's try a simpler approach: check if any comparator is a no-op
    # for all inputs (i.e., the values are always already in order)
    # This is hard to check without simulation, so let's try a different approach.
    
    # Try removing comparators one at a time from the end
    result = list(network)
    for idx in range(len(result) - 1, -1, -1):
        if _time_remaining() < 5:
            break
        candidate = result[:idx] + result[idx+1:]
        if _verify_network(candidate, n):
            result = candidate
            # Reset index since we removed one
            if idx > 0:
                idx += 1  # Re-check the same position after removal
        else:
            # Keep this comparator
            pass
    
    return result

def _try_improve_by_reordering(network, n=16):
    """Try to find a better ordering of the same comparators."""
    # For now, just return the network as-is
    # A more sophisticated approach would use local search
    return network

def construct():
    """Return a sorting network for 16 wires."""
    # Start with Batcher's odd-even mergesort
    net = _batcher_odd_even_mergesort(16)
    
    # Verify correctness
    if not _verify_network(net, 16):
        # Fallback to odd-even transposition if verification fails
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Try to remove redundant comparators
    if _time_remaining() > 10:
        improved = _remove_redundant_comparators(net, 16)
        if len(improved) < len(net):
            net = improved
    
    return net
# EVOLVE-BLOCK-END
