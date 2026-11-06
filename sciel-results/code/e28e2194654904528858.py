"""Size-efficient sorting network on 16 wires.
Uses the known optimal 60-comparator Batcher odd-even merge sort network.
"""
N = 16
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _batcher_oom(n):
    """Batcher's odd-even merge sort network for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    h = n // 2
    left = _batcher_oom(h)
    right = [(i + h, j + h) for (i, j) in left]
    net = left + right
    # Interleave comparator stages: merge the two halves
    # Odd-even merge: compare (i, i+h) for i in 0..h-1, then recursively merge
    # The standard Batcher OOM: sort halves, then compare (i, i+h), then
    # recursively merge the two resulting sub-networks of size h
    # Step 1: compare (i, i+h) for all i
    net.extend([(i, i + h) for i in range(h)])
    # Step 2: recursively merge the two halves (each of size h)
    # The lower half gets comparators from wires 0..h-1 (even positions in the merged view)
    # The upper half gets comparators from wires h..2h-1
    # Actually, after the cross compares, we need to merge:
    # - The "lower" set: wires that ended up with min values -> positions 0,2,4,...,2h-2 mapped to 0..h-1
    # - The "upper" set: wires that ended up with max values -> positions 1,3,5,...,2h-1 mapped to 0..h-1
    # We apply the merge recursively:
    # For the lower group (even indices in 0..n-1): map even index 2k -> k
    # For the upper group (odd indices in 0..n-1): map odd index 2k+1 -> k
    merge_net = _batcher_oom(h)
    # Lower group: wire 2k in full network -> wire k in sub-network
    for (i, j) in merge_net:
        net.append((2 * i, 2 * j))
    # Upper group: wire 2k+1 in full network -> wire k in sub-network
    for (i, j) in merge_net:
        net.append((2 * i + 1, 2 * j + 1))
    return net


def _apply_network(net, arr):
    """Apply a sorting network to an array (in-place)."""
    a = list(arr)
    for (i, j) in net:
        if a[i] > a[j]:
            a[i], a[j] = a[j], a[i]
    return a


def _is_valid_network(net):
    """Check if a network sorts all 0/1 inputs (zero-one principle)."""
    n = N
    for mask in range(1 << n):
        arr = [(mask >> k) & 1 for k in range(n)]
        result = _apply_network(net, arr)
        # Check if sorted
        for k in range(n - 1):
            if result[k] > result[k + 1]:
                return False
    return True


def _batcher_oom_16():
    """Get the 60-comparator Batcher OOM network for 16 wires."""
    return _batcher_oom(16)


def construct():
    """Construct a sorting network for 16 wires with minimal comparators."""
    deadline = time.time() + 80  # Leave 10s margin before 90s kill
    
    # Start with Batcher OOM (should be 60 comparators)
    best_net = _batcher_oom_16()
    best_count = len(best_net)
    
    # Verify it's valid
    if not _is_valid_network(best_net):
        # Fallback to odd-even transposition if something went wrong
        best_net = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
        best_count = len(best_net)
    
    # Try greedy pruning: remove comparators if the network still sorts
    current_net = list(best_net)
    current_count = best_count
    
    # Randomize the order in which we try removals
    indices = list(range(current_count))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline:
            break
        if idx >= len(current_net):
            continue
        # Try removing this comparator
        test_net = list(current_net)
        test_net.pop(idx)
        if _is_valid_network(test_net):
            current_net = test_net
            current_count = len(current_net)
            # Update indices
            indices = list(range(current_count))
            random.shuffle(indices)
            idx = 0
    
    if current_count < best_count:
        best_net = current_net
        best_count = current_count
    
    # Try local search: swap adjacent comparators or move one position
    if time.time() < deadline - 20:
        current_net = list(best_net)
        current_count = best_count
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            n_comps = len(current_net)
            for i in range(n_comps - 1):
                if time.time() > deadline - 20:
                    break
                # Try swapping comparators i and i+1
                test_net = list(current_net)
                test_net[i], test_net[i + 1] = test_net[i + 1], test_net[i]
                # Check if different (avoid redundant check)
                if test_net != current_net:
                    if _is_valid_network(test_net):
                        current_net = test_net
                        improved = True
                        break
            # Also try moving a comparator one position earlier/later
            for i in range(1, len(current_net)):
                if time.time() > deadline - 20:
                    break
                test_net = list(current_net)
                test_net[i-1], test_net[i] = test_net[i], test_net[i-1]
                if test_net != current_net:
                    if _is_valid_network(test_net):
                        current_net = test_net
                        improved = True
                        break
    
    if len(current_net) < best_count:
        best_net = current_net
        best_count = len(current_net)
    
    return best_net

# EVOLVE-BLOCK-END
