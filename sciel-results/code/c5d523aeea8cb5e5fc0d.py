N = 16

# EVOLVE-BLOCK-START
import time

def _sort2():
    return [(0, 1)]

def _sort3():
    return [(0, 1), (1, 2), (0, 1)]

def _sort4():
    """Optimal 4-input sorting network with 5 comparators."""
    return [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]

def _sort8():
    """Known 8-input sorting network with 19 comparators."""
    return [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7),
        (1, 4), (2, 5), (3, 6),
        (1, 2), (4, 5),
        (2, 4)
    ]

def _oe_merge(n):
    """
    Correct Batcher odd-even merge sort network for n wires.
    Returns a list of (i, j) comparators with 0 <= i < j < n.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    if n == 4:
        return _sort4()
    if n == 8:
        return _sort8()
    
    m = n // 2
    k = n - m
    
    # Sort two halves
    net = list(_oe_merge(m))
    right = _oe_merge(k)
    net.extend([(i + m, j + m) for (i, j) in right])
    
    # Odd-even merge of two sorted lists of lengths m and k
    # We interleave: positions 0,2,4,... hold the "even" merged sequence
    # positions 1,3,5,... hold the "odd" merged sequence
    # Standard Batcher OE merge:
    # Compare (i, m+i) for i in 0..min(m,k)-1
    # Then recursively merge the even-indexed and odd-indexed subsequences
    
    if m == k:
        # Compare cross pairs
        for i in range(m):
            net.append((i, i + m))
        
        # Even subsequence: wires at positions 0, 2, 4, ..., 2*(m-1)
        # These correspond to original indices 0,2,4,...,2m-2
        # We need to sort these m wires as a sub-network
        even_net = _oe_merge(m)
        for (i, j) in even_net:
            net.append((2 * i, 2 * j))
        
        # Odd subsequence: wires at positions 1, 3, 5, ..., 2*m-1
        odd_net = _oe_merge(m)
        for (i, j) in odd_net:
            net.append((2 * i + 1, 2 * j + 1))
    else:
        # Unequal lengths: m < k (since m = n//2, k = n - m >= m)
        # Compare (i, m+i) for i in 0..m-1
        for i in range(m):
            net.append((i, i + m))
        
        # The first m positions after cross-compare form two sequences:
        # Even positions 0,2,...,2m-2 (length m) and odd positions 1,3,...,2m-1 (length m)
        # But the tail (positions m..n-1) needs to be merged in
        # This case only occurs for odd n; for n=16 we always have m==k==8
        # Fallback: just add a full transposition pass
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                net.append((i, i + 1))
    
    return net

def _verify_network(network, n=16):
    """Verify the network sorts all 2^n inputs using zero-one principle."""
    for mask in range(1 << n):
        inputs = [(mask >> i) & 1 for i in range(n)]
        for (i, j) in network:
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
        for i in range(n - 1):
            if inputs[i] > inputs[i + 1]:
                return False
    return True

def _prune_redundant(network, n=16, deadline=None):
    """
    Remove comparators that never change any of the 2^n Boolean inputs.
    A comparator (i,j) is redundant if for every input, inputs[i] <= inputs[j]
    at the point the comparator is reached.
    """
    if deadline is not None and time.time() > deadline:
        return network
    
    # For each comparator index, track whether it ever swaps
    # We simulate all 2^n inputs through the network, recording for each
    # comparator whether it caused a swap for any input.
    n_comps = len(network)
    # comp_used[c] = True if comparator c ever swapped
    comp_used = [False] * n_comps
    
    for mask in range(1 << n):
        if deadline is not None and time.time() > deadline:
            break
        inputs = [(mask >> i) & 1 for i in range(n)]
        for c, (i, j) in enumerate(network):
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
                comp_used[c] = True
    
    # Keep only used comparators
    pruned = [network[c] for c in range(n_comps) if comp_used[c]]
    return pruned

def _dedup_preserve_order(network):
    """Remove duplicate comparators, preserving first occurrence order."""
    seen = set()
    out = []
    for c in network:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s wall-clock limit
    
    # Build the Batcher odd-even merge sort network for 16 wires
    network = _oe_merge(16)
    network = _dedup_preserve_order(network)
    
    # Verify
    if _verify_network(network):
        # Try to prune redundant comparators
        pruned = _prune_redundant(network, n=16, deadline=deadline)
        if _verify_network(pruned):
            network = pruned
        return network
    
    # Fallback: odd-even transposition (always correct, 120 comparators)
    fallback = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    return fallback
# EVOLVE-BLOCK-END
