"""Size-efficient sorting networks on 18 wires."""
import time
import random

N = 18
SEED_COUNT = 153  # odd-even transposition network comparator count
HARD_DEADLINE = 85.0  # seconds, safety margin before 90s kill


def _odd_even_transposition_network(n):
    """Generate the odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
    return comps


def _bitonic_merge(n, start=0, reverse=False):
    """Generate comparators for a bitonic merge on wires [start, start+n)."""
    comps = []
    if n <= 1:
        return comps
    half = n // 2
    comps.extend(_bitonic_merge(half, start, reverse))
    comps.extend(_bitonic_merge(half, start + half, reverse))
    for i in range(half):
        a = start + i
        b = start + n - 1 - i
        if reverse:
            a, b = b, a
        if a < b:
            comps.append((a, b))
        else:
            comps.append((b, a))
    return comps


def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires (power of 2)."""
    comps = []
    if n <= 1:
        return comps
    half = n // 2
    comps.extend(_bitonic_sort_network(half))
    comps.extend(_bitonic_sort_network(half))
    comps.extend(_bitonic_merge(n))
    return comps


def _verify_network_bitparallel(comps, n=N):
    """
    Verify that the network sorts all Boolean inputs using bit-parallelism.
    Returns True if correct, False otherwise.
    """
    if not comps:
        return True
    
    # Precompute comparator operations
    comp_ops = []
    for i, j in comps:
        if i > j:
            i, j = j, i
        mask_i = 1 << i
        mask_j = 1 << j
        comp_ops.append((i, j, mask_i, mask_j))
    
    for mask in range(1 << n):
        wires = mask
        for i, j, mask_i, mask_j in comp_ops:
            bit_i = (wires & mask_i) != 0
            bit_j = (wires & mask_j) != 0
            if bit_i and not bit_j:
                # Swap: set j, clear i
                wires |= mask_j
                wires &= ~mask_i
        # Check if sorted: all 0s before all 1s
        # A sorted boolean array has form [0,...,0,1,...,1]
        # So the mask must be of form (2^k - 1) << (n - k) for some k
        # Quick check: find where 0s end and 1s begin
        # If wires is sorted, there's a point where all lower bits are 0 and all upper are 1
        # Or it's all 0s or all 1s
        # Simple check: for each i < j, if bit i > bit j, not sorted
        for i in range(n - 1):
            if ((wires >> i) & 1) > ((wires >> (i + 1)) & 1):
                return False
    return True


def _verify_network_sampled(comps, n=N, sample_size=10000):
    """Verify with random samples for quick checks."""
    for _ in range(sample_size):
        wires = random.randint(0, (1 << n) - 1)
        for i, j in comps:
            if i > j:
                i, j = j, i
            bit_i = (wires >> i) & 1
            bit_j = (wires >> j) & 1
            if bit_i > bit_j:
                wires ^= (1 << i)
                wires ^= (1 << j)
        for i in range(n - 1):
            if ((wires >> i) & 1) > ((wires >> (i + 1)) & 1):
                return False
    return True


def _prune_comparators(comps, n=N, deadline=None):
    """
    Try to remove redundant comparators from the network.
    Uses iterative removal with verification.
    """
    comps = list(comps)
    improved = True
    while improved:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if deadline and time.time() > deadline:
                return comps
            # Try removing comparator at idx
            candidate = comps[:idx] + comps[idx + 1:]
            if _verify_network_bitparallel(candidate, n):
                comps = candidate
                improved = True
                break  # Restart from beginning since list changed
    return comps


def _local_search(comps, n=N, deadline=None, max_iterations=500):
    """
    Local search to improve the network by trying random modifications.
    """
    comps = list(comps)
    best = comps[:]
    best_len = len(comps)
    
    for iteration in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        if len(comps) >= best_len:
            break
        
        # Try random modifications
        mod_type = random.random()
        
        if mod_type < 0.6 and comps:
            # Try removing a random comparator
            idx = random.randint(0, len(comps) - 1)
            candidate = comps[:idx] + comps[idx + 1:]
            if _verify_network_bitparallel(candidate, n):
                comps = candidate
                best = comps[:]
                best_len = len(comps)
        elif mod_type < 0.85 and len(comps) >= 2:
            # Try swapping two comparators
            idx = random.randint(0, len(comps) - 2)
            candidate = comps[:]
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            if _verify_network_bitparallel(candidate, n):
                comps = candidate
        else:
            # Try adding a random comparator
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            pos = random.randint(0, len(comps))
            candidate = comps[:pos] + [(i, j)] + comps[pos:]
            if _verify_network_bitparallel(candidate, n):
                comps = candidate
                # This might enable further pruning
                if time.time() < deadline - 10:
                    pruned = _prune_comparators(comps, n, deadline)
                    if len(pruned) < len(comps):
                        comps = pruned
                        best = comps[:]
                        best_len = len(comps)
    
    return best


def _construct_initial_network():
    """
    Construct an initial sorting network for N=18 wires.
    Uses a combination of known structures.
    """
    # Start with odd-even transposition (153 comparators)
    # Then try to build something better
    
    # For 18 wires, we can use:
    # - Bitonic sort on 32 wires (padded), then filter
    # - Or a custom construction
    
    # Let's use a known good construction:
    # Split into groups and use bitonic merge
    
    # Actually, let's start with odd-even transposition and prune
    return _odd_even_transposition_network(N)


def construct():
    """
    Construct a compare-exchange network that sorts all 18-input sequences.
    Returns ordered list of integer pairs (i, j), 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + HARD_DEADLINE
    
    # Start with odd-even transposition network
    comparators = _construct_initial_network()
    
    # Verify it works
    if not _verify_network_bitparallel(comparators, N):
        # This shouldn't happen, but fallback
        return comparators
    
    # Try to prune redundant comparators
    if time.time() < deadline - 5:
        comparators = _prune_comparators(comparators, N, deadline)
    
    # Try local search for further improvements
    if time.time() < deadline - 5:
        comparators = _local_search(comparators, N, deadline, max_iterations=200)
    
    # Final verification
    if not _verify_network_bitparallel(comparators, N):
        # Fallback to odd-even transposition
        comparators = _odd_even_transposition_network(N)
    
    return comparators
