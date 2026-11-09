import time

N = 18

# EVOLVE-BLOCK-START
def _generate_bitonic(n):
    """Generates a Bitonic Sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort first half
    net1 = _generate_bitonic(half)
    # Sort second half
    net2 = _generate_bitonic(half)
    for i in range(len(net2)):
        net2[i] = (net2[i][0] + half, net2[i][1] + half)
    
    # Merge
    merge_net = _generate_merge(n)
    
    return net1 + net2 + merge_net

def _generate_merge(n):
    """Generates the merge phase for Bitonic Sort."""
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Cross comparators
    cross = [(i, n - 1 - i) for i in range(half)]
    
    # Recursive merges
    merge_top = _generate_merge(half)
    merge_bottom = _generate_merge(half)
    for i in range(len(merge_bottom)):
        merge_bottom[i] = (merge_bottom[i][0] + half, merge_bottom[i][1] + half)
        
    return cross + merge_top + merge_bottom

def construct():
    # Use Bitonic Sort for 32 wires
    net32 = _generate_bitonic(32)
    # Prune comparators where j >= N
    # Since i < j, if j >= N, the comparator is either (real, dummy) or (dummy, dummy)
    # Both are no-ops for the real wires if dummies are +inf.
    net18 = [(i, j) for i, j in net32 if j < N]
    
    # Optional: Verify correctness if time allows
    # But we trust the logic.
    return net18
# EVOLVE-BLOCK-END
