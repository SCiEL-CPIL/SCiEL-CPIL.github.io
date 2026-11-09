"""Rewritten program for size-efficient sorting networks on 18 wires."""
import time
import random
import itertools
from collections import deque

N = 18
SEED_COUNT = 153  # odd-even transposition network comparator count
DEADLINE_MARGIN = 10.0  # seconds before hard kill
HARD_DEADLINE = 90.0


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
    """
    Generate comparators for a bitonic merge on wires [start, start+n).
    If reverse, the input is assumed to be in reverse bitonic order.
    """
    comps = []
    if n <= 1:
        return comps
    # Recursively merge halves
    comps.extend(_bitonic_merge(n // 2, start, reverse))
    comps.extend(_bitonic_merge(n // 2, start + n // 2, reverse))
    # Final merge phase
    for i in range(n // 2):
        a = start + i
        b = start + n - 1 - i
        if reverse:
            a, b = b, a
        comps.append((a, b))
    return comps


def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires (power of 2)."""
    comps = []
    if n <= 1:
        return comps
    # Recursively sort halves
    comps.extend(_bitonic_sort_network(n // 2))
    comps.extend(_bitonic_sort_network(n // 2))
    # Merge
    comps.extend(_bitonic_merge(n))
    return comps


def _odd_even_merge(n, start=0, reverse=False):
    """
    Generate comparators for an odd-even merge on n wires starting at 'start'.
    n must be even.
    """
    comps = []
    if n <= 2:
        if n == 2:
            if reverse:
                comps.append((start + 1, start))
            else:
                comps.append((start, start + 1))
        return comps
    # Split into two halves
    half = n // 2
    # Recursively merge each half
    comps.extend(_odd_even_merge(half, start, reverse))
    comps.extend(_odd_even_merge(half, start + half, reverse))
    # Interleave and merge
    for i in range(half - 1):
        # Compare (start + 2*i + 1) with (start + half + 2*i)
        a = start + 2 * i + 1
        b = start + half + 2 * i
        if reverse:
            a, b = b, a
        comps.append((a, b))
    return comps


def _odd_even_sort_network(n, start=0, reverse=False):
    """Generate an odd-even sorting network for n wires."""
    comps = []
    if n <= 1:
        return comps
    if n == 2:
        if reverse:
            comps.append((start + 1, start))
        else:
            comps.append((start, start + 1))
        return comps
    # Split into two halves
    half = n // 2
    # Recursively sort each half
    comps.extend(_odd_even_sort_network(half, start, reverse))
    comps.extend(_odd_even_sort_network(half, start + half, reverse))
    # Merge
    comps.extend(_odd_even_merge(n, start, reverse))
    return comps


def _verify_network(comps, n=N, sample_size=262144):
    """
    Verify that the network sorts all Boolean inputs.
    Returns True if correct, False otherwise.
    """
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comps:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check if sorted
        for i in range(n - 1):
            if wires[i] > wires[i + 1]:
                return False
    return True


def _verify_network_fast(comps, n=N):
    """
    Faster verification using bit-level parallelism.
    Represents each wire as a bit in an integer, where bit i represents wire i.
    """
    # Precompute masks for each comparator
    comp_masks = []
    for i, j in comps:
        mask_i = 1 << i
        mask_j = 1 << j
        comp_masks.append((mask_i, mask_j))
    
    for mask in range(1 << n):
        wires = mask
        for mask_i, mask_j in comp_masks:
            # Extract bits
            bit_i = (wires >> i) & 1 if False else ((wires & mask_i) != 0)
            bit_j = ((wires & mask_j) != 0)
            if bit_i and not bit_j:
                # Swap: set j, clear i
                wires |= mask_j
                wires &= ~mask_i
        # Check if sorted: all 1s should be in lower bits
        # For sorted, we need: if bit i is 0 and bit j is 1 for i < j, that's wrong
        # Actually for sorted ascending: lower indices have smaller values
        # So 0s should come before 1s
        # Check: the mask should be of form 0...01...1
        # i.e., there exists k such that bits 0..k-1 are 0 and bits k..n-1 are 1
        # Or all 0s or all 1s
        # Quick check: count trailing zeros, then verify the rest are all 1s
        # Actually simpler: check that the mask is a "prefix of zeros" pattern
        # A sorted boolean array has the form [0,0,...,0,1,1,...,1]
        # So the mask must be of the form (2^k - 1) << (n - k) for some k, or 0, or (2^n - 1)
        # Let's just check directly
        for i in range(n - 1):
            if ((wires >> i) & 1) > ((wires >> (i + 1)) & 1):
                return False
    return True


def _prune_comparators(comps, n=N, deadline=None):
    """
    Try to remove redundant comparators from the network.
    Returns a pruned list of comparators.
    """
    comps = list(comps)
    improved = True
    while improved:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if deadline and time.time() > deadline:
                break
            # Try removing comparator at idx
            candidate = comps[:idx] + comps[idx + 1:]
            if _verify_network_fast(candidate, n):
                comps = candidate
                improved = True
                break  # Restart from beginning since list changed
        if deadline and time.time() > deadline:
            break
    return comps


def _local_search(comps, n=N, deadline=None, max_iterations=1000):
    """
    Local search to improve the network by swapping/adding/removing comparators.
    """
    comps = list(comps)
    best = comps
    best_len = len(comps)
    
    for iteration in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        if len(comps) >= best_len:
            break
        
        # Try random modifications
        # 1. Try removing a random comparator
        if comps and random.random() < 0.5:
            idx = random.randint(0, len(comps) - 1)
            candidate = comps[:idx] + comps[idx + 1:]
            if _verify_network_fast(candidate, n):
                comps = candidate
                best = list(comps)
                best_len = len(comps)
        # 2. Try swapping two adjacent comparators
        elif len(comps) >= 2:
            idx = random.randint(0, len(comps) - 2)
            candidate = comps[:]
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            if _verify_network_fast(candidate, n):
                # Swapping doesn't change length, but might enable further pruning
                # Only accept if it doesn't break anything (it won't if both are valid)
                # Actually swapping might break validity, so we check
                pass  # We already checked, so accept
                comps = candidate
        # 3. Try adding a random comparator
        elif random.random() < 0.3:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            candidate = comps + [(i, j)]
            if _verify_network_fast(candidate, n):
                # Adding a comparator that keeps it valid - not useful for size reduction
                # But might enable pruning elsewhere
                pass
    
    return best


def construct():
    """
    Construct a sorting network for N=18 wires.
    Returns an ordered list of integer pairs (i, j) where 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + HARD_DEADLINE - DEADLINE_MARGIN
    
    # Start with a known good network
    # Option 1: Odd-even transposition (153 comparators)
    # Option 2: Bitonic sort for 32 wires, then restrict to 18 (might be wasteful)
    # Option 3: Odd-even sort for 18 wires
    
    # Let's use odd-even transposition as seed
    network = _odd_even_transposition_network(N)
    original_len = len(network)
    
    # Verify it's correct
    assert _verify_network_fast(network, N), "Seed network failed verification"
    
    # Try pruning
    pruned = _prune_comparators(network, N, deadline=deadline)
    network = pruned
    current_len = len(network)
    
    if time.time() > deadline:
        return network
    
    # Try local search for further improvements
    improved = _local_search(network, N, deadline=deadline, max_iterations=200)
    network = improved
    current_len = len(network)
    
    # Final verification
    assert _verify_network_fast(network, N), "Final network failed verification"
    
    return network


# Ensure the module is self-contained and the function is accessible
if __name__ == "__main__":
    result = construct()
    print(f"Constructed network with {len(result)} comparators")
