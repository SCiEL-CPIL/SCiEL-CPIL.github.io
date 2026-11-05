"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16
ALL_MASKS = (1 << N) - 1

# EVOLVE-BLOCK-START
def _apply_network(comparators, mask):
    """Apply compare-exchange network to a bitmask. Returns sorted bitmask."""
    for i, j in comparators:
        # min to i, max to j
        vi = (mask >> i) & 1
        vj = (mask >> j) & 1
        if vi > vj:
            # swap: set i to 0, j to 1
            mask &= ~(1 << i)
            mask |= (1 << j)
    return mask

def _verify_network(comparators):
    """Verify network sorts all 2^N boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        if _apply_network(comparators, mask) != mask:
            return False
    return True

def _verify_network_fast(comparators):
    """Faster verification using bit-parallel where possible."""
    # For N=16, 65536 checks. Each check applies len(comparators) ops.
    # Total ops ~ 65536 * len(comparators). For 60 comparators: ~4M ops.
    # In Python this might be slow. Let's use a smarter approach.
    # Actually, let's just iterate but with early exit.
    for mask in range(1 << N):
        m = mask
        for i, j in comparators:
            vi = (m >> i) & 1
            vj = (m >> j) & 1
            if vi > vj:
                m &= ~(1 << i)
                m |= (1 << j)
        if m != mask:
            return False
    return True

def _odd_even_transposition():
    """Generate odd-even transposition network (120 comparators)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _akl_network():
    """
    AKL (Akers, Kroning, Loos) or similar known 16-wire sorting network.
    Let's use a known good construction.
    
    A well-known 16-wire sorting network has 60 comparators.
    One such network is based on the Batcher odd-even merge sort or
    a specific hand-optimized network.
    
    Let's try to construct one using a recursive Batcher approach first,
    then try to optimize.
    """
    # Batcher's odd-even merge sort for 16 elements
    # This typically uses more comparators (around 84-100+).
    # Let's use a known 60-comparator network if we can recall/construct it.
    
    # For now, let's use a heuristic construction.
    # Actually, let's start with the 120 comparator network and try to reduce.
    return _odd_even_transposition()

def _local_search(comparators, max_time=85.0):
    """
    Local search to reduce comparator count.
    Strategy: Try removing each comparator and check if network still sorts.
    If it does, keep it removed. Repeat until no more can be removed.
    Also try swapping/reordering comparators to enable further removals.
    """
    start_time = time.time()
    best = list(comparators)
    improved = True
    
    while improved:
        improved = False
        for idx in range(len(best)):
            if time.time() - start_time > max_time:
                return best
            # Try removing comparator at idx
            candidate = best[:idx] + best[idx+1:]
            # Verify
            if _verify_network_fast(candidate):
                best = candidate
                improved = True
                break  # Restart from beginning since network changed
    
    return best

def construct():
    """
    Construct a sorting network for 16 wires.
    Uses local search to minimize comparator count.
    """
    start_time = time.time()
    
    # Start with odd-even transposition (120 comparators)
    network = _odd_even_transposition()
    
    # Local search to remove redundant comparators
    network = _local_search(network, max_time=85.0)
    
    return network
# EVOLVE-BLOCK-END
