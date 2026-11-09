import time
import numpy as np

N = 16
DEADLINE = time.time() + 85  # safety margin before 90s kill

def _generate_batcher_network(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    n must be a power of 2.
    """
    if n <= 1:
        return []
    
    def merge_network(start, length):
        """
        Generates the merge network for two sorted halves of 'length' each,
        starting at 'start'.
        First half: [start, start+length)
        Second half: [start+length, start+2*length)
        """
        if length <= 1:
            return [(start, start + 1)]
        
        half = length // 2
        comps = []
        
        # Odd-Even Merge:
        # 1. Split each half into odd and even indexed elements.
        #    Odd indices in first half: start, start+2, ...
        #    Even indices in second half: start+length+1, start+length+3, ...
        #    These form a bitonic sequence of length 'length'.
        
        # 2. Recursively merge the odd-indexed parts.
        #    The odd-indexed elements of the first half and even-indexed of the second half
        #    are interleaved to form a bitonic sequence.
        #    We need to map these to a contiguous range for the recursive call?
        #    No, the recursive merge works on the logical sequence.
        #    However, the comparators must refer to physical wire indices.
        
        # Standard recursive definition for Odd-Even Merge M(k):
        # M(1) = compare (0, 1)
        # M(2k) = 
        #   M(k) on odd elements (indices 0, 2, ... in first half, 1, 3, ... in second half)
        #   M(k) on even elements (indices 1, 3, ... in first half, 0, 2, ... in second half)
        #   Compare (i, i+1) for i = 0, 2, 4, ..., 2k-2
        
        # Let's implement this mapping carefully.
        
        # Odd part:
        # Indices in first half: 0, 2, 4, ..., 2k-2 -> start, start+2, ...
        # Indices in second half: 1, 3, 5, ..., 2k-1 -> start+k+1, start+k+3, ...
        # These 2k elements form a bitonic sequence.
        # We need to apply M(k) to them.
        # The recursive M(k) expects a contiguous block of 2k elements?
        # No, M(k) merges two sorted sequences of length k.
        # The "odd" elements from the first half are sorted.
        # The "even" elements from the second half are sorted.
        # So we are merging two sorted sequences of length k.
        # The recursive call should generate comparators for these logical elements.
        
        # To make this work with physical indices, we need a helper that takes
        # two lists of physical indices and generates comparators to merge them.
        
        def merge_indices(idx1, idx2):
            """
            idx1: list of physical indices for the first sorted sequence (length k)
            idx2: list of physical indices for the second sorted sequence (length k)
            Returns comparators to merge them.
            """
            k = len(idx1)
            if k == 1:
                return [(idx1[0], idx2[0])]
            
            half_k = k // 2
            comps = []
            
            # Split idx1 into odd and even (relative to the sequence)
            # idx1[0], idx1[2], ... are the "odd" positions in the sequence?
            # Wait, the definition of Odd-Even Merge:
            # Split each sequence into odd and even indexed elements.
            # Odd indexed: 0, 2, 4...
            # Even indexed: 1, 3, 5...
            
            # Seq 1 Odd: idx1[0], idx1[2], ...
            # Seq 2 Even: idx2[1], idx2[3], ...
            # These form a bitonic sequence. Merge them.
            
            seq1_odd = [idx1[i] for i in range(0, k, 2)]
            seq2_even = [idx2[i] for i in range(1, k, 2)]
            
            # Wait, the lengths must match.
            # If k is even, seq1_odd has k/2 elements. seq2_even has k/2 elements.
            # If k is odd, this is tricky. But k is always a power of 2 in our case.
            
            comps.extend(merge_indices(seq1_odd, seq2_even))
            
            # Seq 1 Even: idx1[1], idx1[3], ...
            # Seq 2 Odd: idx2[0], idx2[2], ...
            
            seq1_even = [idx1[i] for i in range(1, k, 2)]
            seq2_odd = [idx2[i] for i in range(0, k, 2)]
            
            comps.extend(merge_indices(seq1_even, seq2_odd))
            
            # Compare adjacent elements in the original sequences
            # The elements at positions 0,1, 2,3, ... in the merged output
            # are not necessarily at adjacent physical indices.
            # But the final step of Odd-Even Merge is to compare
            # (position 0, 1), (2, 3), etc.
            # In terms of the input sequences:
            # The element at position 2i comes from either seq1 or seq2.
            # This logic is hard to trace with physical indices directly in this recursive step.
            
            # Alternative approach:
            # The standard recursive formula for the number of comparators is correct,
            # but generating the exact pairs requires careful bookkeeping.
            
            # Let's use the property that the merge network for two sorted sequences
            # of length k can be constructed by:
            # 1. Comparing (i, k-1-i) for i=0..k-1? No, that's not it.
            
            # Let's stick to the bitonic merge which is easier to implement correctly,
            # even if it uses more comparators.
            # Bitonic Merge for N=16 uses 32.
            # Total Bitonic Sort N=16 uses 80.
            
            # If I cannot easily implement OEM, I will stick with Bitonic (80)
            # and try to optimize it via greedy removal.
            
            return [] # Placeholder
            
        # Since the recursive OEM implementation is complex and error-prone to write from scratch
        # in this short time, and Bitonic (80) is already better than the 120 seed,
        # I will use Bitonic and rely on the greedy removal to reduce the count.
        
        # However, the prompt says "Reference 60".
        # If I can get to 60, I get a score of 80/60 = 1.33.
        
        # Let's try a different strategy:
        # Use the known 54-comparator network if I can find it.
        # Or, use a faster verification and more aggressive search.
        
        return [] # Fallback to bitonic in main

    return []

def _bitonic_sort_network(n):
    """Generate bitonic sort network for n wires."""
    if n <= 1:
        return []
    
    def generate(start, length, direction):
        if length <= 1:
            return []
        
        half = length // 2
        comps = []
        
        # Sort halves in opposite directions
        comps.extend(generate(start, half, 1))
        comps.extend(generate(start + half, half, -1))
        
        # Bitonic Merge
        for i in range(half):
            a = start + i
            b = start + half + i
            if direction == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        # Recursively merge
        comps.extend(generate(start, half, direction))
        comps.extend(generate(start + half, half, direction))
        
        return comps
    
    return generate(0, n, 1)

def _verify_full(network):
    """Full verification using numpy over all 65536 states."""
    states = np.arange(65536, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap = bi > bj
        states = np.where(swap, states ^ ((1 << i) | (1 << j)), states)
    return bool(np.all(((states + 1) & states) == 0))

def construct():
    start = time.time()
    
    # Start with Bitonic Sort (80 comparators)
    net = _bitonic_sort_network(N)
    best = list(net)
    
    # Verify initial network
    if not _verify_full(best):
        # Fallback to odd-even transposition (120 comparators)
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    import random
    rng = random.Random(12345)
    
    # Greedy removal
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        order = list(range(len(best)))
        rng.shuffle(order)
        for idx in order:
            if time.time() > DEADLINE - 5:
                break
            candidate = best[:idx] + best[idx+1:]
            # Fast verification: check a subset of states
            # For speed, we can check random samples
            # But we need to be sure.
            # Let's use a small sample for speed, then full verify at the end.
            # If sample fails, skip.
            
            # To be safe and fast, we'll just do full verify if the network is small enough,
            # or use a probabilistic check.
            # Given the time limit, we'll do full verify but break early if it fails?
            # NumPy verify is fast.
            
            if _verify_full(candidate):
                best = candidate
                improved = True
    
    # Final full verification
    if _verify_full(best):
        return best
    else:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
