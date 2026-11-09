<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # 1. Generate initial network: Odd-Even Transposition (120 comparators)
    # This is a known correct sorting network.
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))

    # 2. Generate all 2^N zero-one inputs for verification
    # Using bit operations for speed
    num_inputs = 1 << N
    # Create an array of indices 0 to 2^N - 1
    indices = np.arange(num_inputs, dtype=np.uint32)
    # Expand bits to create the input matrix
    # inputs[i, j] is the j-th bit of index i
    # We want (i, j) to be the value of wire j for input i.
    # Standard binary representation: bit 0 is LSB.
    # So inputs[i, j] = (i >> j) & 1
    inputs = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1

    # 3. Helper function to check if a network is sorting using 0-1 principle
    # This uses NumPy vectorization to process all 65536 inputs simultaneously.
    def is_sorting(net):
        # Copy inputs to state
        state = inputs.copy()
        
        # Apply each comparator
        for i, j in net:
            # min goes to i, max goes to j
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        
        # Check if all rows are sorted (non-decreasing)
        # A row is sorted if state[:, k] <= state[:, k+1] for all k
        # We check if there is any violation: state[:, k] > state[:, k+1]
        # If any violation exists, it's not sorting.
        # We can check all adjacent pairs at once.
        # diff = state[:, 1:] - state[:, :-1]
        # If any diff < 0, not sorted.
        # Since values are 0 or 1, diff is in {-1, 0, 1}.
        # So we just check if any diff == -1.
        # Or simply: np.any(state[:, :-1] > state[:, 1:])
        return not np.any(state[:, :-1] > state[:, 1:])

    # Verify initial network (should be true)
    if not is_sorting(network):
        # Fallback: return the known correct network
        return network

    # 4. Greedy Removal Search
    # Try to remove comparators one by one.
    # We iterate through the network in reverse order (often easier to remove later ones first? 
    # Actually, order doesn't matter much for greedy, but reverse might be slightly faster 
    # if we modify the list in place or rebuild it).
    # We'll use a list and rebuild or pop.
    # Since we need to check the whole network for each removal, 
    # the cost is O(len(net)) checks per removal attempt.
    # Total cost is roughly O(M^2 * C) where M is number of comparators and C is cost of is_sorting.
    # C is ~65536 * 16 operations in NumPy, which is fast.
    # M starts at 120. 120^2 = 14400 checks.
    # Each check is ~1-2 ms? Maybe less.
    # 14400 * 2ms = 28 seconds. This might be tight but feasible.
    # We can optimize by only checking if the comparator is "potentially" redundant.
    # But let's try the direct approach first with a time guard.

    best_network = network[:]
    best_len = len(network)
    
    # We'll try multiple passes. In each pass, we try to remove any comparator.
    # We stop when no more can be removed or time runs out.
    
    max_passes = 5 # Safety limit
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        changed = False
        # Iterate backwards to avoid index shifting issues if we remove
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Check if candidate is still sorting
            if is_sorting(candidate):
                # It is sorting! Remove it.
                network = candidate
                changed = True
                # Update best
                if len(network) < best_len:
                    best_len = len(network)
                    best_network = network[:]
                
        if not changed:
            break

    # 5. Optional: Try to swap or reorder? 
    # Reordering doesn't change the count.
    # Swapping might allow further removals.
    # But greedy removal is usually quite effective.
    
    # Final check on best_network
    if is_sorting(best_network):
        return best_network
    else:
        # Should not happen, but fallback to original
        return network

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _batcher_oe_merge_sort(n):
    """Generate Batcher's odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    mid = n // 2
    # Recursively sort each half
    left = _batcher_oe_merge_sort(mid)
    right = [(i + mid, j + mid) for i, j in _batcher_oe_merge_sort(n - mid)]
    
    # Merge the two sorted halves using odd-even merge
    def oe_merge(a, b, offset_a=0, offset_b=0):
        """Merge two sorted sequences of lengths a and b."""
        if a == 0:
            return [(offset_b + i, offset_b + i + 1) for i in range(b - 1) if b > 1]
        if b == 0:
            return [(offset_a + i, offset_a + i + 1) for i in range(a - 1) if a > 1]
        
        if a == 1 and b == 1:
            return [(offset_a, offset_b)]
        
        if a == 1:
            # Single element from left, merge with b elements from right
            result = []
            # Compare left[0] with right[0], right[1], ..., right[b-1]
            # But we need to be careful about the structure
            # Actually for a=1, we just compare the single element with appropriate right elements
            # Standard odd-even merge for a=1:
            # Compare (offset_a, offset_b + 0) if b >= 1
            # Then recursively merge
            if b >= 1:
                result.append((offset_a, offset_b))
                # Now the larger of the two goes to offset_b position
                # We need to merge the remaining
                # This is getting complex, let me use a different approach
            # Let me use the standard recursive definition
            
        # Standard odd-even merge:
        # If both have even length:
        #   Compare (a[0], b[0]), (a[1], b[1]), ...
        #   Then recursively merge odd-indexed and even-indexed results
        # If one has odd length:
        #   Handle the extra element separately
        
        # Let me implement it properly
        result = []
        
        if a % 2 == 0 and b % 2 == 0:
            # Both even: compare corresponding elements
            for i in range(min(a, b)):
                result.append((offset_a + i, offset_b + i))
            # Now recursively merge
            # The results of comparisons go into interleaved positions
            # This is complex to do in-place, so let me use a different approach
            
        # Actually, let me use a simpler recursive formulation
        # Odd-even merge of sequences A (length a) and B (length b):
        # 1. If a == 1: compare A[0] with B[0], then recursively merge
        #    the result with B[1:]
        # 2. If b == 1: compare A[0] with B[0], then recursively merge
        #    A[1:] with the result
        # 3. If both > 1: 
        #    - Compare A[0] with B[0], A[1] with B[1], etc.
        #    - Split results into odd and even indexed
        #    - Recursively merge odd-indexed and even-indexed
        
        # This is getting too complex for inline generation. Let me use a known implementation.
        return result
    
    # For simplicity, let me just use a known good starting network
    # Batcher's bitonic sort is easier to generate
    return left + right

def _bitonic_sort(n):
    """Generate Batcher's bitonic sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    def bitonic_merge(n, offset=0, ascending=True):
        """Merge a bitonic sequence of length n starting at offset."""
        if n <= 1:
            return []
        
        result = []
        # Compare elements that are n/2 apart
        half = n // 2
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if not ascending:
                a, b = b, a
            result.append((a, b))
        
        # Recursively merge the two halves
        left = bitonic_merge(half, offset, ascending)
        right = bitonic_merge(half, offset + half, ascending)
        
        return result + left + right
    
    def bitonic_sort_rec(n, offset=0, ascending=True):
        """Sort a bitonic sequence of length n starting at offset."""
        if n <= 1:
            return []
        
        half = n // 2
        # Sort first half ascending, second half descending (or vice versa)
        left = bitonic_sort_rec(half, offset, ascending)
        right = bitonic_sort_rec(half, offset + half, not ascending)
        
        # Merge
        merge = bitonic_merge(n, offset, ascending)
        
        return left + right + merge
    
    return bitonic_sort_rec(n, 0, True)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Generate all 2^N zero-one inputs for verification
    num_inputs = 1 << N
    indices = np.arange(num_inputs, dtype=np.uint32)
    inputs = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1

    def is_sorting(net):
        state = inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        return not np.any(state[:, :-1] > state[:, 1:])

    # Start with Batcher's bitonic sort (~72 comparators for N=16)
    network = _bitonic_sort(N)
    
    if not is_sorting(network):
        # Fallback to odd-even transposition
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
    
    best_network = network[:]
    best_len = len(network)
    
    # Local search: greedy removal with random restarts and reordering
    def greedy_remove(net, deadline):
        """Try to remove redundant comparators."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = net[:idx] + net[idx+1:]
                if is_sorting(candidate):
                    net = candidate
                    changed = True
                    if len(net) < best_len:
                        nonlocal best_len, best_network
                        best_len = len(net)
                        best_network = net[:]
        return net
    
    # First pass: greedy removal from bitonic sort
    network = greedy_remove(network, deadline)
    
    # Multiple random restarts with different initial perturbations
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_restarts = max(1, int(remaining_time / 20))  # ~20s per restart
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Start from current best
            net = best_network[:]
            
            # Randomly swap some comparators (reordering might enable further removals)
            for _ in range(min(5, len(net))):
                if len(net) < 2:
                    break
                i = random.randint(0, len(net) - 2)
                j = random.randint(i + 1, len(net) - 1)
                net[i], net[j] = net[j], net[i]
            
            # Try greedy removal
            net = greedy_remove(net, deadline)
            
            # Try random comparator swaps that maintain sorting
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                if len(net) < 2:
                    break
                i = random.randint(0, len(net) - 2)
                j = random.randint(i + 1, len(net) - 1)
                candidate = net[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if is_sorting(candidate):
                    net = candidate
            
            # Final greedy removal
            net = greedy_remove(net, deadline)
    
    # Final verification
    if is_sorting(best_network):
        return best_network
    else:
        # Fallback
        return network

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
