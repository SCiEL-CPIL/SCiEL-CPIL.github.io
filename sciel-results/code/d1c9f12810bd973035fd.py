"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START

def _batcher_merge(left_start, right_start, size):
    """Generate odd-even merge network to merge two sorted lists of 'size' elements.
    left_start: starting wire index for left list
    right_start: starting wire index for right list
    Returns list of (i, j) comparators.
    """
    if size == 1:
        return []
    
    comparators = []
    # Odd-even merge: first merge odd positions, then even positions
    # Then recursively merge the odd sub-list and even sub-list
    
    # First phase: compare (left[i], right[i]) for all i
    # Actually, odd-even merge works differently:
    # It's defined recursively. For merging two sorted lists A and B of length n:
    # 1. Compare A[0] with B[0], A[1] with B[1], ..., A[n-1] with B[n-1] (n comparators)
    # 2. This gives two sorted sub-sequences: one from odd-indexed results, one from even-indexed
    # 3. Recursively merge those
    
    # Standard odd-even merge:
    # Phase 1: n comparators pairing (A[i], B[i])
    for i in range(size):
        comparators.append((left_start + i, right_start + i))
    
    # Now we need to merge the "even" and "odd" subsequences
    # The even subsequence: min(A[0],B[0]), max(A[1],B[1]), ... wait no.
    
    # Let me use the standard recursive definition:
    # odd-even merge(A, B, n):
    #   if n == 1: return
    #   Compare A[0] with B[0], A[1] with B[1], ..., A[n-1] with B[n-1]
    #   Now the even positions form a sorted list, odd positions form a sorted list
    #   Recursively odd-even merge the even list and odd list
    
    # Actually the standard way: after comparing A[i] with B[i] for all i,
    # the values at even indices (0, 2, 4, ...) form a sorted sequence and
    # values at odd indices (1, 3, 5, ...) form a sorted sequence.
    # Then recursively merge those.
    
    # But this requires the lists to be interleaved in memory. Let me use a different approach.
    
    # Actually, let me just build the network using the known recursive structure.
    # For merging two sorted lists of length n stored in consecutive wires:
    # Left list: wires [left_start, left_start+n)
    # Right list: wires [right_start, right_start+n)
    
    # The odd-even merge network:
    # Step 1: n comparators: (left_start+i, right_start+i) for i in 0..n-1
    # Step 2: The result has even-indexed positions sorted and odd-indexed positions sorted
    #         But they're interleaved. We need to extract and re-merge.
    
    # This is getting complex. Let me use a simpler approach: just hardcode a known good network.
    # Or use a different construction.
    
    return comparators


def _build_batcher(n):
    """Build Batcher's odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    comparators = []
    
    # Recursively sort each half
    left_net = _build_batcher(half)
    for (a, b) in left_net:
        comparators.append((a, b))
    
    right_net = _build_batcher(half)
    for (a, b) in right_net:
        comparators.append((half + a, half + b))
    
    # Merge the two sorted halves using odd-even merge
    # The odd-even merge of two sorted lists of length m uses 2m-1 comparators
    # Let me implement it properly
    
    # Odd-even merge for two lists of length m:
    # The lists are in wires [0, m) and [m, 2m)
    # We need to produce a merged sorted list in [0, 2m)
    
    def odd_even_merge(m, offset=0):
        """Merge two sorted lists of length m starting at offset and offset+m.
        Returns list of comparators (relative to offset).
        """
        if m == 1:
            return []
        
        comps = []
        # Compare corresponding elements
        for i in range(m):
            comps.append((offset + i, offset + m + i))
        
        # Now even-indexed results are sorted, odd-indexed results are sorted
        # Even indices: 0, 2, 4, ..., 2m-2 (m elements)
        # Odd indices: 1, 3, 5, ..., 2m-1 (m elements)
        # Recursively merge the even sub-list and odd sub-list
        
        # The even sub-list is in wires: offset+0, offset+2, offset+4, ...
        # The odd sub-list is in wires: offset+1, offset+3, offset+5, ...
        
        # We need to recursively merge these, but they're not contiguous.
        # This is the tricky part. In a standard implementation, we'd need to
        # think about this differently.
        
        # Alternative: use the fact that after the first phase, we have:
        # wire offset+2i contains min(A[i], B[i]) for the "even" subsequence
        # wire offset+2i+1 contains max(A[i], B[i]) for the "odd" subsequence
        # But these aren't quite right either.
        
        # Let me use a different, cleaner recursive definition.
        # Actually, the standard odd-even merge works on interleaved data.
        # Let me just use a known construction.
        
        return comps
    
    # This is getting complicated. Let me just use a simple but correct approach:
    # Use Batcher's bitonic sort instead, which is easier to implement.
    
    def bitonic_sort(n, offset=0):
        """Bitonic sort network for n wires starting at offset."""
        if n <= 1:
            return []
        
        comps = []
        
        # Split into two halves
        half = n // 2
        
        # Recursively sort first half ascending, second half descending
        left_comps = bitonic_sort(half, offset)
        comps.extend(left_comps)
        
        # For the right half, we need to sort it in descending order
        # We can do this by sorting ascending and then reversing the comparator directions
        right_comps = bitonic_sort(half, offset + half)
        for (a, b) in right_comps:
            comps.append((b, a))  # Reverse direction to sort descending
        
        # Merge the bitonic sequence
        comps.extend(bitonic_merge(n, offset))
        
        return comps
    
    def bitonic_merge(n, offset=0):
        """Merge a bitonic sequence of length n starting at offset."""
        if n <= 1:
            return []
        
        comps = []
        half = n // 2
        
        # Compare elements i and i+half
        for i in range(half):
            comps.append((offset + i, offset + i + half))
        
        # Recursively merge the two halves
        left_comps = bitonic_merge(half, offset)
        comps.extend(left_comps)
        right_comps = bitonic_merge(half, offset + half)
        comps.extend(right_comps)
        
        return comps
    
    # Use bitonic sort
    return bitonic_sort(n)


def construct():
    """Construct a sorting network for 16 wires."""
    # Use bitonic sort which is straightforward to implement
    # Bitonic sort for n=16 uses n*(log2(n)+1)/2 * log2(n) comparators
    # For n=16: 16*(4+1)/2 * 4 = 16*5/2*4 = 160? That's worse than odd-even.
    
    # Let me just implement Batcher's odd-even merge sort properly.
    # 
    # The key insight: odd-even merge of two sorted lists A[0..m-1] and B[0..m-1]:
    # 1. Compare A[i] with B[i] for all i (m comparators)
    # 2. The result: even-indexed positions form a sorted list, odd-indexed form a sorted list
    # 3. Recursively merge the even list (length m) and odd list (length m)
    #
    # But the even and odd lists are interleaved in memory. To handle this,
    # we can think of it as: after step 1, wire 2i has min(A[i],B[i]) and wire 2i+1 has max(A[i],B[i]).
    # The even subsequence (wires 0,2,4,...) is sorted, odd subsequence (wires 1,3,5,...) is sorted.
    # We need to merge these two subsequences, which are not contiguous.
    #
    # One way: just do the recursive merge on the interleaved structure.
    
    def odd_even_merge_network(m, base=0):
        """
        Generate comparators to merge two sorted lists of length m.
        List A is in wires [base, base+m)
        List B is in wires [base+m, base+2m)
        After the network, wires [base, base+2m) contain the merged sorted list.
        """
        if m == 1:
            return [(base, base + 1)]
        
        comps = []
        
        # Phase 1: compare A[i] with B[i]
        for i in range(m):
            comps.append((base + i, base + m + i))
        
        # Now, the even-indexed wires (base+0, base+2, base+4, ...) contain a sorted sequence
        # and odd-indexed wires (base+1, base+3, base+5, ...) contain a sorted sequence.
        # We need to merge these two subsequences.
        # 
        # The even subsequence has m elements in wires: base+0, base+2, ..., base+2m-2
        # The odd subsequence has m elements in wires: base+1, base+3, ..., base+2m-1
        #
        # To recursively merge them, we can think of it as:
        # - The "even" list is stored at positions 0, 2, 4, ... relative to base
        # - The "odd" list is stored at positions 1, 3, 5, ... relative to base
        #
        # We can recursively call odd_even_merge_network on these, but they're not contiguous.
        # 
        # Alternative approach: just generate the comparators directly.
        # The recursive structure means:
        # - Merge the even subsequence with itself? No.
        #
        # Let me think again. After phase 1:
        # Wire base+2i: min(A[i], B[i]) for i=0..m-1, and this sequence is sorted
        # Wire base+2i+1: max(A[i], B[i]) for i=0..m-1, and this sequence is sorted
        #
        # Now we need to merge the sequence [min(A[0],B[0]), min(A[1],B[1]), ...] with
        # [max(A[0],B[0]), max(A[1],B[1]), ...]
        #
        # These are interleaved. The standard trick is to note that this is equivalent to
        # an odd-even merge of two sorted lists of length m, where the lists are stored
        # in alternating positions.
        #
        # For the recursive call, we can define a helper that merges two sorted lists
        # stored in arbitrary positions.
        
        # Let me use a different, cleaner approach: define the merge in terms of
        # the logical positions.
        
        # Actually, let me just hardcode a known-good 16-wire network with ~60 comparators.
        # Based on known results, a good 16-wire sorting network exists with 60 comparators.
        
        # Let me try a different strategy: start with a known construction and verify.
        
        # Batcher's odd-even merge sort for n=16:
        # The number of comparators is: C(16) = 2*C(8) + M(8)
        # where M(8) is the number of comparators to merge two sorted 8-lists
        # M(8) = 2*M(4) + 8 = 2*(2*M(2)+4) + 8 = 2*(2*3+4)+8 = 2*10+8 = 28
        # Wait: M(2) = 3 (compare 2 pairs, then merge the results)
        # M(2): compare A[0] with B[0], A[1] with B[1] (2 comparators), then the even and odd
        #       subsequences each have length 1, so no more comparators. Total: 2? 
        #       But we need 3 for merging two sorted 2-lists.
        #
        # Let me just count: to merge two sorted lists of length 2:
        # [a0,a1] and [b0,b1] where a0<=a1, b0<=b1
        # Compare a0 with b0, a1 with b1: (a0,b0), (a1,b1)
        # Now even positions: min(a0,b0), min(a1,b1) - but is this sorted? Not necessarily!
        # We also need to compare the cross terms.
        #
        # The correct odd-even merge for m=2:
        # 1. Compare (a0,b0) and (a1,b1): 2 comparators
        # 2. Now wire0=min(a0,b0), wire1=max(a0,b0), wire2=min(a1,b1), wire3=max(a1,b1)
        #    Even subsequence: wire0, wire2 = min(a0,b0), min(a1,b1) - sorted? 
        #    Odd subsequence: wire1, wire3 = max(a0,b0), max(a1,b1) - sorted?
        # 3. Need to merge even subsequence with odd subsequence: they're interleaved
        #    Compare wire0 with wire1, wire2 with wire3: 2 more comparators
        #    Then compare wire1 with wire2: 1 more comparator
        #    Total: 2+2+1 = 5? That's too many.
        #
        # I think I'm overcomplicating this. Let me just use a brute-force search or
        # a known construction.
        
        # Given the time constraint, let me use a simpler approach:
        # Use the odd-even transposition network but try to optimize it.
        # Or just return a known good network.
        
        return comps
    
    # Given the complexity, let me just use a simple correct network and try to improve.
    # Start with odd-even transposition (120 comparators) and try local search.
    
    deadline = time.time() + 80  # 80 second budget
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    def simulate(network, input_vec):
        """Simulate the sorting network on a given input vector."""
        state = list(input_vec)
        for (i, j) in network:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        return state
    
    def is_sorted(vec):
        return all(vec[i] <= vec[i+1] for i in range(len(vec)-1))
    
    def verify_network(network):
        """Verify the network sorts all 2^16 Boolean inputs using zero-one principle."""
        for mask in range(1 << N):
            vec = [(mask >> i) & 1 for i in range(N)]
            result = simulate(network, vec)
            if not is_sorted(result):
                return False, vec
        return True, None
    
    # Verify the initial network
    valid, _ = verify_network(network)
    if not valid:
        # Shouldn't happen for odd-even transposition
        return network
    
    best_network = network
    best_count = len(network)
    
    # Try to remove comparators one at a time
    for _ in range(500):  # Try up to 500 removals
        if time.time() > deadline - 5:
            break
        
        removed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            valid, _ = verify_network(test_network)
            if valid:
                network = test_network
                removed = True
                break
        
        if not removed:
            break
    
    best_network = network
    best_count = len(network)
    
    # Try to swap comparator positions to enable more removals
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for idx in range(len(network) - 1):
            if time.time() > deadline - 5:
                break
            
            # Try swapping network[idx] and network[idx+1]
            test_network = network[:]
            test_network[idx], test_network[idx+1] = test_network[idx+1], test_network[idx]
            
            # Try removing a random comparator from the swapped network
            for rem_idx in range(len(test_network) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                test2 = test_network[:rem_idx] + test_network[rem_idx+1:]
                valid, _ = verify_network(test2)
                if valid:
                    network = test2
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return network

# EVOLVE-BLOCK-END

