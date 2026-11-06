import time
import numpy as np

N = 18
NUM_BOOLS = 1 << N
DEADLINE = 85.0  # 85s limit to be safe

# Precompute sorted mask
_SORTED_MASK = np.zeros(NUM_BOOLS, dtype=np.bool_)
for _k in range(N + 1):
    _val = ((1 << _k) - 1) << (N - _k)
    _SORTED_MASK[_val] = True

# Precompute masks for wires
_CLEAR_MASKS = [~(1 << i) for i in range(N)]
_SET_MASKS = [(1 << i) for i in range(N)]

def _verify_network(comparators):
    """Verify that the network sorts all 2^N Boolean inputs."""
    if not comparators:
        return False
    current = np.arange(NUM_BOOLS, dtype=np.int32)
    for i, j in comparators:
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        mask_swap = (bit_i == 1) & (bit_j == 0)
        if np.any(mask_swap):
            current = np.where(mask_swap, (current & _CLEAR_MASKS[i]) | _SET_MASKS[j], current)
    return np.all(_SORTED_MASK[current])

def _batcher_oe_merge_sort(n, offset=0):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    comps = []
    
    # Recursively sort the first half and second half
    half = n // 2
    comps.extend(_batcher_oe_merge_sort(half, offset))
    comps.extend(_batcher_oe_merge_sort(n - half, offset + half))
    
    # Merge the two sorted halves
    # The merge step of Odd-Even Merge Sort
    # It consists of phases.
    # Phase 1: Compare (0, 1), (2, 3), ... (even, odd)
    # Phase 2: Compare (1, 2), (3, 4), ... (odd, even)
    # ...
    # This is effectively a butterfly network.
    
    # Standard Odd-Even Merge for two sorted lists of size k:
    # The merge of two sorted lists of length k takes k * ceil(log2(k)) / 2 comparators?
    # Actually, the recursive structure is:
    # Merge(A, B) where A and B are sorted.
    # If len(A) == len(B) == k:
    #   Split A into A_even, A_odd
    #   Split B into B_even, B_odd
    #   Recursively Merge(A_even, B_odd)
    #   Recursively Merge(A_odd, B_even)
    #   Then compare (0,1), (2,3)...
    
    # Let's implement the merge step specifically for equal sizes first, then handle odd.
    # For simplicity and correctness, we can use a generic merge function.
    
    def merge_sorted(A_start, B_start, size):
        """
        Merges two sorted sequences of length 'size' starting at A_start and B_start.
        Assumes the sequences are already sorted.
        Returns the comparators for the merge step.
        """
        if size == 0:
            return []
        if size == 1:
            # Compare the single elements
            return [(A_start, B_start)]
        
        comps = []
        half_size = size // 2
        
        # Split into even and odd indexed elements
        # A_even: A_start, A_start+2, ...
        # A_odd: A_start+1, A_start+3, ...
        # B_even: B_start, B_start+2, ...
        # B_odd: B_start+1, B_start+3, ...
        
        # Recursive merges
        # Merge A_even with B_odd
        # Indices: A_even[i] is at A_start + 2*i
        #           B_odd[i] is at B_start + 2*i + 1
        # This is not a contiguous block, so we can't just call merge_sorted directly on offsets.
        # We need to map the virtual indices to physical wires.
        
        # This recursive definition is tricky with physical offsets.
        # Alternative: Just use Batcher's standard Merge Sort which is easier to implement with offsets.
        # Standard Batcher's Merge Sort:
        # Sort left, Sort right.
        # Merge:
        #   For k in range(log2(n)):
        #     for i in range(n // 2 ** (k+1)):
        #        ...
        
        # Let's stick to a simpler, known-good construction: Batcher's Merge Sort (not Odd-Even).
        # It's slightly larger but easier to get right.
        # For N=18, we can split 9+9.
        
        return [] # Placeholder, we will replace this logic

    # Let's use a different approach for the seed:
    # Use a precomputed small network for 18 wires if possible?
    # Or just use a "Bitonic Sort" network.
    # Bitonic Sort for N=18?
    # Bitonic sort requires N to be a power of 2 for the standard implementation.
    # We can pad to 32 wires? That would be huge.
    
    # Let's use a "Shellsort" like network? No.
    
    # Let's implement Batcher's Merge Sort for N=18 properly.
    # Split 18 into 9 and 9.
    # Sort 9s.
    # Merge 9s.
    
    # How to sort 9?
    # Split 9 into 4 and 5.
    # Sort 4, Sort 5.
    # Merge 4 and 5.
    
    # This recursive structure is fine.
    
    def generate_batcher_merge(n, offset=0):
        if n <= 1:
            return []
        
        comps = []
        left_size = n // 2
        right_size = n - left_size
        
        # Sort left and right
        comps.extend(generate_batcher_merge(left_size, offset))
        comps.extend(generate_batcher_merge(right_size, offset + left_size))
        
        # Merge left and right
        # Batcher's Merge of two sorted lists of size L and R
        # The merge step consists of phases.
        # Phase 1: Compare (0, R), (1, R+1), ... (L-1, R+L-1) ? No.
        
        # Standard Batcher Merge:
        # The merge of two sorted arrays of size n/2 each takes n log2(n) / 2 comparators.
        # The structure is:
        # for k in range(int(np.log2(n))):
        #    step = 1 << k
        #    for i in range(0, n, 2 * step):
        #       for j in range(step):
        #          # Compare (offset + i + j, offset + i + j + step)
        #          # But we are merging two separate arrays.
        
        # Let's look up the exact comparator pattern for Batcher's Merge.
        # It's easier to just use a known library or implementation.
        # Since I can't import external libs, I will implement a simple "Odd-Even Transposition" on the merged part?
        # No, that's inefficient.
        
        # Let's use a simpler seed: **Knuth's Sorting Network** or just **Odd-Even Transposition** but optimized?
        # The previous program used OET (153).
        # Can we do better than OET easily?
        # Yes, Batcher's Merge Sort for 16 is 69.
        # For 18, we can do:
        # 1. Sort 16 wires using Batcher's Merge Sort (69 comps).
        # 2. The remaining 2 wires are unsorted.
        # 3. We need to merge the sorted 16 with the 2 unsorted wires.
        # This is complex.
        
        # Alternative Seed: **Batcher's Odd-Even Merge Sort** for N=18.
        # I will implement the recursive Odd-Even Merge carefully.
        
        return comps

    # Let's implement a verified Batcher's Merge Sort for N=18.
    # I will use a standard implementation found in literature.
    
    def _merge_batcher(left_start, right_start, size):
        """
        Merges two sorted blocks of 'size' starting at left_start and right_start.
        Returns comparators.
        """
        if size == 0:
            return []
        if size == 1:
            return [(left_start, right_start)]
        
        comps = []
        # The merge of two sorted lists of size 'size'
        # can be done by:
        # 1. Splitting each into even and odd indexed elements.
        # 2. Recursively merging left_even with right_odd, and left_odd with right_even.
        # 3. Comparing adjacent pairs.
        
        # This is the Odd-Even Merge.
        # left_even indices: left_start, left_start+2, ...
        # left_odd indices: left_start+1, left_start+3, ...
        
        # To call _merge_batcher recursively, we need contiguous blocks?
        # No, Odd-Even Merge works on interleaved indices.
        # This makes it hard to implement with simple offsets.
        
        # Let's fall back to a simpler, slightly larger network:
        # **Batcher's Merge Sort** (not Odd-Even).
        # It is easier to implement with contiguous blocks.
        
        # Merge step for two sorted blocks of size 'size' (where total is 2*size):
        # The comparators are:
        # for k in range(int(np.log2(size)) + 1):
        #    step = 1 << k
        #    for i in range(0, size, 2 * step):
        #       for j in range(step):
        #          # Compare (left_start + i + j, right_start + i + j) ??
        
        # Actually, let's just use a precomputed network for N=18 if I can find one in my head?
        # No.
        
        # Let's use a "Greedy" construction?
        # Start with OET (153).
        # Run the greedy removal.
        # The previous program's greedy removal was slow because it restarted.
        # My new program will run greedy removal more efficiently.
        # Even if it doesn't reach 77, it should reach ~80-90 quickly.
        
        return comps

    # Let's just use the OET seed but with a MUCH better removal algorithm.
    # The OET seed is 153.
    # If we can remove 50 comparators, we get 103.
    # If we can remove 70, we get 83.
    
    # To make removal faster:
    # 1. Verify network.
    # 2. Iterate through comparators.
    # 3. If a comparator is redundant, remove it.
    # 4. Do NOT restart the index. Just continue.
    # 5. Repeat the pass until no more removals.
    
    return comps

def construct():
    start_time = time.time()
    
    # Seed: Odd-Even Transposition
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
            
    # Verify baseline
    if not _verify_network(comparators):
        return comparators
        
    # Greedy Removal
    improved = True
    while improved and time.time() < start_time + DEADLINE:
        improved = False
        # Iterate backwards
        for idx in range(len(comparators) - 1, -1, -1):
            if time.time() > start_time + DEADLINE:
                break
            
            # Try removing
            trial = comparators[:idx] + comparators[idx+1:]
            if _verify_network(trial):
                comparators = trial
                improved = True
                # Do not break, continue to check for more redundancies in the same pass
                # This is faster than restarting
                
    return comparators
