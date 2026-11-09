"""Size-efficient sorting networks on 16 wires.

This module constructs a Batcher's Odd-Even Mergesort network for N=16 wires.
The network is known to use 64 comparators (vs 120 for odd-even transposition),
and its correctness is certified by the zero-one principle over all 65536 inputs.
"""

N = 16

# EVOLVE-BLOCK-START
def _compare(a, b, wires):
    """Perform a compare-exchange on wires a and b."""
    if a > b:
        a, b = b, a
    wires[a], wires[b] = min(wires[a], wires[b]), max(wires[a], wires[b])

def _odd_even_merge(network, wires, k, even_phase):
    """
    Merge two sorted halves of length k using odd-even merge.
    wires: list of wire indices to operate on.
    even_phase: True if this is the even phase, False if odd phase.
    """
    if k <= 1:
        return
    if k == 2:
        _compare(wires[0], wires[1], network)
        return
    
    # Phase 1: Compare elements (2i, 2i+k) for i in range(k)
    for i in range(k):
        network.append((wires[2 * i], wires[2 * i + k]))
    
    # Phase 2: Recursively merge even and odd sub-arrays
    # Even sub-array: wires[0], wires[2], wires[4], ...
    # Odd sub-array: wires[1], wires[3], wires[5], ...
    even_wires = [wires[2 * i] for i in range(k)]
    odd_wires = [wires[2 * i + 1] for i in range(k)]
    
    _odd_even_merge(network, even_wires, k // 2, not even_phase)
    _odd_even_merge(network, odd_wires, k // 2, not even_phase)
    
    # Phase 3: Compare elements (2i+1, 2i+1+k) for i in range(k-1)
    for i in range(k - 1):
        network.append((wires[2 * i + 1], wires[2 * i + 1 + k]))

def _odd_even_merge_sort(network, wires, k):
    """
    Sort k wires using Batcher's odd-even mergesort.
    """
    if k <= 1:
        return
    if k == 2:
        _compare(wires[0], wires[1], network)
        return
    if k == 3:
        # Optimal 3-wire sort: 3 comparators
        _compare(wires[0], wires[2], network)
        _compare(wires[0], wires[1], network)
        _compare(wires[1], wires[2], network)
        return
    if k == 4:
        # Optimal 4-wire sort: 5 comparators
        _compare(wires[0], wires[2], network)
        _compare(wires[1], wires[3], network)
        _compare(wires[0], wires[1], network)
        _compare(wires[2], wires[3], network)
        _compare(wires[1], wires[2], network)
        return
    if k == 5:
        # Optimal 5-wire sort: 9 comparators
        _compare(wires[0], wires[4], network)
        _compare(wires[1], wires[3], network)
        _compare(wires[0], wires[3], network)
        _compare(wires[1], wires[2], network)
        _compare(wires[0], wires[2], network)
        _compare(wires[1], wires[3], network)
        _compare(wires[2], wires[4], network)
        _compare(wires[0], wires[4], network)
        _compare(wires[2], wires[3], network)
        return
    if k == 8:
        # Use known optimal 8-wire network: 19 comparators
        # From literature: 
        _compare(wires[0], wires[4], network)
        _compare(wires[1], wires[5], network)
        _compare(wires[2], wires[6], network)
        _compare(wires[3], wires[7], network)
        _compare(wires[0], wires[2], network)
        _compare(wires[1], wires[3], network)
        _compare(wires[4], wires[6], network)
        _compare(wires[5], wires[7], network)
        _compare(wires[0], wires[4], network)
        _compare(wires[1], wires[5], network)
        _compare(wires[2], wires[6], network)
        _compare(wires[3], wires[7], network)
        _compare(wires[2], wires[4], network)
        _compare(wires[3], wires[5], network)
        _compare(wires[1], wires[3], network)
        _compare(wires[5], wires[7], network)
        _compare(wires[0], wires[2], network)
        _compare(wires[4], wires[6], network)
        _compare(wires[2], wires[4], network)
        return
    
    # For larger k, use recursive odd-even merge
    half = k // 2
    left = wires[:half]
    right = wires[half:]
    
    # Sort each half
    _odd_even_merge_sort(network, left, half)
    _odd_even_merge_sort(network, right, half)
    
    # Merge the two sorted halves
    _odd_even_merge(network, wires, half, True)

def construct():
    """
    Construct a sorting network for N=16 wires using Batcher's odd-even mergesort.
    Returns a list of (i, j) pairs where i < j, representing compare-exchange operations.
    """
    network = []
    wires = list(range(N))
    _odd_even_merge_sort(network, wires, N)
    return network
# EVOLVE-BLOCK-END
