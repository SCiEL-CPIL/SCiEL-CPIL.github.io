"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a state tuple."""
    lst = list(state)
    if lst[i] > lst[j]:
        lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)

def _verify_network(comparators):
    """Verify sorting network using zero-one principle over all 2^N Boolean inputs."""
    n = N
    for mask in range(1 << n):
        state = tuple(1 if (mask >> i) & 1 else 0 for i in range(n))
        for i, j in comparators:
            if state[i] > state[j]:
                state = state[:i] + (state[j],) + state[i+1:j] + (state[i],) + state[j+1:]
        if any(state[i] > state[i+1] for i in range(n - 1)):
            return False
    return True

def _compute_depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    # Build layers: two comparators can be in the same layer if they don't share wires
    depth = 0
    remaining = list(comparators)
    while remaining:
        # Greedily assign comparators to current layer
        layer = []
        used_wires = set()
        still_remaining = []
        for comp in remaining:
            i, j = comp
            if i not in used_wires and j not in used_wires:
                layer.append(comp)
                used_wires.add(i)
                used_wires.add(j)
            else:
                still_remaining.append(comp)
        depth += 1
        remaining = still_remaining
    return depth

def _greedy_construct(deadline):
    """
    Global structural search: start with empty network, add comparators greedily
    to fix the most "critical" violations across Boolean inputs.
    """
    start_time = time.time()
    n = N
    
    # Precompute all Boolean inputs as bitmasks
    all_inputs = list(range(1 << n))
    
    # Track which inputs are still unsorted
    # For each input, we track the current state after applying known comparators
    # We'll use a different approach: maintain for each input the "deficit" - 
    # how many inversions remain
    
    # Better approach: use a set of "critical" comparators identified by analysis
    # For now, let's use a heuristic: try to build a network that's close to optimal
    
    # Known good construction: use a hybrid approach
    # Start with a partial network and refine
    
    # Let's try a divide-and-conquer construction similar to Batcher's odd-even merge
    # but optimized for 16 wires
    
    # Batcher's bitonic sort for 16 elements uses 56 comparators
    # Let's construct it properly
    
    comparators = []
    
    # Helper: bitonic merge for 2^k elements
    def bitonic_merge(n, offset, direction):
        """Generate comparators for bitonic merge of n elements starting at offset."""
        comps = []
        # For bitonic merge of n elements, we need log2(n) stages
        k = n // 2
        while k > 0:
            for i in range(0, n, 2 * k):
                for j in range(k):
                    a = offset + i + j
                    b = offset + i + k + j
                    if direction == 'up':
                        comps.append((a, b))
                    else:
                        comps.append((b, a))
            k //= 2
        return comps
    
    def bitonic_sort(n, offset, direction):
        """Generate comparators for bitonic sort of n elements starting at offset."""
        if n <= 1:
            return []
        comps = []
        half = n // 2
        # Sort first half ascending, second half descending
        comps.extend(bitonic_sort(half, offset, 'up'))
        comps.extend(bitonic_sort(half, offset + half, 'down'))
        # Merge
        comps.extend(bitonic_merge(n, offset, direction))
        return comps
    
    # Standard bitonic sort: sort in ascending order
    # First create bitonic sequences, then merge
    # For 16 elements: 
    # Stage 1: sort pairs (8 comparators)
    # Stage 2: merge pairs into 4-groups (16 comparators) 
    # Stage 3: merge 4-groups into 8-groups (32 comparators)
    # Stage 4: merge 8-groups into 16-group (32 comparators)
    
    # Actually, let me use the standard bitonic sort construction
    # which uses n*log2(n)*log2(n)/4 comparators = 16*4*4/4 = 64 for n=16
    
    # Let me build it step by step
    # Level 1: compare adjacent pairs
    for i in range(0, n, 2):
        comparators.append((i, i + 1))
    
    # Level 2: merge pairs into 4-element bitonic sequences
    for base in range(0, n, 4):
        # Compare (base, base+2) and (base+1, base+3)
        comparators.append((base, base + 2))
        comparators.append((base + 1, base + 3))
        # Then the merge within each 4-group
        # For 4 elements, bitonic merge needs 2 stages
        # Stage 1: compare (0,2) and (1,3) - already done above
        # Stage 2: compare (0,1) and (2,3) - but wait, we need to be careful
    
    # Let me restart with a cleaner bitonic sort implementation
    comparators = []
    
    def bitonic_sort_recursive(n, offset, direction):
        """
        Generate comparators for bitonic sort.
        direction: 'up' for ascending, 'down' for descending
        """
        if n <= 1:
            return []
        half = n // 2
        comps = []
        # Sort first half in opposite direction, second half in same direction
        # to create a bitonic sequence
        if direction == 'up':
            comps.extend(bitonic_sort_recursive(half, offset, 'down'))
            comps.extend(bitonic_sort_recursive(half, offset + half, 'up'))
        else:
            comps.extend(bitonic_sort_recursive(half, offset, 'up'))
            comps.extend(bitonic_sort_recursive(half, offset + half, 'down'))
        # Now merge the bitonic sequence
        # Bitonic merge of n elements takes log2(n) stages
        k = n // 2
        while k > 0:
            for i in range(0, n, 2 * k):
                for j in range(k):
                    a = offset + i + j
                    b = offset + i + k + j
                    if direction == 'up':
                        comps.append((a, b))
                    else:
                        comps.append((b, a))
            k //= 2
        return comps
    
    comparators = bitonic_sort_recursive(n, 0, 'up')
    
    # Verify and return
    if _verify_network(comparators):
        return comparators
    
    # Fallback to odd-even transposition if verification fails
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    deadline = time.time() + 85  # Safety margin: 85s out of 90s
    try:
        result = _greedy_construct(deadline)
        return result
    except Exception:
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
