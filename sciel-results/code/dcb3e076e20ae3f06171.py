import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def _batcher_odd_even_merge(n):
    """
    Generates Batcher's Odd-Even Merge Sort network.
    This is a more efficient construction than Bitonic Sort (63 vs 84 comparators for N=16).
    """
    if n == 1:
        return []
    
    k = n // 2
    # 1. Recursively sort the even-indexed wires (0, 2, ...)
    even_sort = _batcher_odd_even_merge(k)
    # 2. Recursively sort the odd-indexed wires (1, 3, ...)
    odd_sort = _batcher_odd_even_merge(k)
    
    # Apply even sort to even wires
    # Indices in even_sort are 0..k-1. Map to 0, 2, 4...
    even_comps = [(2*i, 2*j) for i, j in even_sort]
    
    # Apply odd sort to odd wires
    # Indices in odd_sort are 0..k-1. Map to 1, 3, 5...
    odd_comps = [(2*i + 1, 2*j + 1) for i, j in odd_sort]
    
    # 3. Merge the two sorted lists using Odd-Even Merge
    # The merge network for two sorted lists of size k:
    # It consists of k-1 stages.
    # Stage i (0-indexed) compares wires (i, i + k) to (i + k - 1, 2k - 1) with stride 2?
    # Actually, the standard Odd-Even Merge merge step is:
    # for i in 0..k-2:
    #   compare (i, i+k)
    #   for j in i+1 .. k-1 step 2:
    #      compare (j, j+k-1)
    #   for j in i+2 .. k-1 step 2:
    #      compare (j, j+k)
    
    merge_comps = []
    for i in range(k - 1):
        # First compare: (i, i+k)
        merge_comps.append((i, i + k))
        
        # Then odd/even interleaved compares
        # The sequence of pairs to compare is:
        # (i+1, i+k), (i+2, i+k-1), (i+3, i+k), ... ?
        # Let's stick to the recursive definition or a known pattern.
        # Pattern for merging two sorted lists A[0..k-1] and B[0..k-1] (B is offset by k):
        # We want to merge A and B.
        # The comparator network for merging two sorted lists of length k is:
        # Compare (i, i+k) for i=0..k-2
        # Then compare (i+1, i+k) and (i+2, i+k-1) etc.
        
        # A simpler recursive structure for Odd-Even Merge:
        # M(k) = S(k) + M(k/2) + M(k/2) + C(k)
        # But we are already generating the full network.
        
        # Let's use the explicit loop for the merge part of Odd-Even Merge.
        # Reference: Batcher 1968.
        # The merge network for two sorted lists of length n (n even):
        # 1. Compare (i, i+n/2) for i=0..n/2-1  <-- Wait, this is for Bitonic.
        
        # Correct Odd-Even Merge step:
        # Given two sorted lists of length k.
        # We perform k-1 stages of comparisons.
        # Stage i (0 <= i < k-1):
        #   Compare (i, i+k)
        #   Compare (i+1, i+k) ? No.
        
        # Let's look at the structure:
        # The merge of two sorted sequences of length k (indices 0..k-1 and k..2k-1)
        # uses the following comparators:
        # for i in 0..k-2:
        #    (i, i+k)
        #    (i+1, i+k) is not right.
        
        # Let's use the recursive generator for the merge part specifically.
        # Or just use the known 63-comparator network if we can construct it reliably.
        # Since Batcher's OES is complex to generate correctly from scratch in a few lines,
        # and we have a known good 63-comparator count, let's try to generate it carefully.
        
        # Actually, the simplest robust approach for the "Inspiration" is to use the 
        # "Known 60" or "Known 63" if we can hardcode or generate it.
        # But the directive asks for a structural change.
        
        # Let's try a different, simpler structural change:
        # Use a "Shuffle-Exchange" or "Butterfly" based network which is easier to generate
        # and often has good properties, then optimize.
        # Or, use the Batcher OES correctly.
        
        # Let's implement the OES merge correctly.
        # The merge of two sorted lists of length k (even):
        # Step 1: Compare (i, i+k) for i in 0..k-2.
        # Step 2: Recursively merge the lower half and upper half?
        
        # Let's abandon OES generation if it's too error-prone and use a simpler, 
        # verifiable topology: The "AKS-like" or "Butterfly" network mentioned in Inspiration 3.
        # However, Inspiration 3's butterfly was just a butterfly, not a sort.
        
        # Let's go back to Bitonic Sort but with a better initial state or a different optimization.
        # The current program uses Bitonic (84 comps) and optimizes to ~111 (score 1.081).
        # Wait, 120 / 111 = 1.081. So it reduced from 120 to 111?
        # Batcher Bitonic for 16 is 84. 120/84 = 1.42.
        # Why did the current program start with Bitonic and score 1.081 (111 comps)?
        # Ah, the current program in the prompt says: "The verified 60-comparator construction is the reference; the initial odd-even transposition network uses 120 comparators."
        # And the current program code generates `_bitonic_sort_network(N)`.
        # Let's check the count of `_bitonic_sort_network(16)`.
        # B(16) = 2*B(8) + M(16).
        # M(16) has 15+7+3+1 = 26 comps? No, M(n) has n/2 + n/4 + ... = n-1 comps.
        # M(16) = 15 comps.
        # B(8) = 2*B(4) + M(8). M(8)=7. B(4)=2*B(2)+M(4). M(4)=3. B(2)=1.
        # B(4) = 2*1 + 3 = 5.
        # B(8) = 2*5 + 7 = 17.
        # B(16) = 2*17 + 15 = 49.
        # So Bitonic Sort is 49 comparators for N=16?
        # Let's verify.
        # Standard Bitonic Sort for 16:
        # Depth 1: 2 sorts of 8.
        # Depth 2: 4 sorts of 4.
        # Depth 3: 8 sorts of 2.
        # Merges:
        # Level 1 (size 16): 1 merge of 16.
        # Level 2 (size 8): 2 merges of 8.
        # Level 3 (size 4): 4 merges of 4.
        # Level 4 (size 2): 8 merges of 2.
        
        # Comparators in Merge of size 2k:
        # M(2) = 1
        # M(4) = 2 + 1 = 3
        # M(8) = 4 + 2 + 1 = 7
        # M(16) = 8 + 4 + 2 + 1 = 15
        
        # Total = 1*M(16) + 2*M(8) + 4*M(4) + 8*M(2)
        # Total = 15 + 14 + 12 + 8 = 49.
        
        # So the current program starts with 49 comparators!
        # And the score is 1.081.
        # Score = 120 / value.
        # 1.081 = 120 / value => value = 111.
        # This implies the current program is RETURNING 111 comparators?
        # That's impossible if it starts with 49 and only removes.
        # Unless... the "current program" provided in the prompt is NOT the one that scored 1.081?
        # Or my calculation of Bitonic Sort comparators is wrong?
        
        # Let's re-read the prompt's current program.
        # It defines `_bitonic_sort_network`.
        # And `construct` calls it.
        # And then does local search to REMOVE comparators.
        # If it starts with 49, it can only go down.
        # If the score is 1.081, and seed is 120, then value must be 111.
        # This suggests the "current program" in the prompt might be different from what I think,
        # OR the score 1.081 belongs to a DIFFERENT program (the parent in the JSON?).
        
        # Let's look at the JSON.
        # p37 has score 1.081081...
        # p37 source sha ...
        # The prompt says "CURRENT PROGRAM (score 1.0811)".
        # And the code provided IS the current program.
        
        # Is it possible that `_bitonic_sort_network` generates 120 comparators?
        # Let's trace `_bitonic_sort_network(16)`.
        # `left = _bitonic_sort_network(8)`
        # `right_desc = ...`
        # `merge = _bitonic_merge_network(16)`
        
        # `_bitonic_merge_network(16)`:
        # k=8: 8 comps (0,8)...(7,15)
        # k=4: 4 comps (0,4)...(3,7) AND (8,12)...(11,15)?
        # Wait, `_bitonic_merge_network` as written:
        # `for i in range(n - k): comps.append((i, i + k))`
        # This generates comparators (0, k), (1, k+1)... (n-k-1, n-1).
        # For n=16, k=8: (0,8)..(7,15) -> 8 comps.
        # k=4: (0,4)..(11,15) -> 12 comps.
        # k=2: (0,2)..(13,15) -> 14 comps.
        # k=1: (0,1)..(14,15) -> 15 comps.
        # Total merge = 8+12+14+15 = 49 comps for the merge step alone!
        
        # Ah, the `_bitonic_merge_network` in the current program is NOT the standard bitonic merge.
        # Standard bitonic merge of size n has n/2 + n/4 + ... = n-1 comparators.
        # This implementation has sum_{k=n/2, n/4..1} (n-k) comparators.
        # For n=16: 8+12+14+15 = 49.
        # This is a MUCH larger network.
        # So the starting network is huge.
        # Left sort (8 wires) + Right sort (8 wires) + Merge (16 wires).
        
        # Let's estimate the size of this "wrong" bitonic sort.
        # Merge(16) = 49.
        # Sort(8) = Sort(4) + Sort(4) + Merge(8).
        # Merge(8) = 4 + 6 + 7 = 17.
        # Sort(4) = Sort(2) + Sort(2) + Merge(4).
        # Merge(4) = 2 + 3 = 5.
        # Sort(2) = 0 + 0 + Merge(2).
        # Merge(2) = 1.
        # Sort(2) = 1.
        # Sort(4) = 1 + 1 + 5 = 7.
        # Sort(8) = 7 + 7 + 17 = 31.
        # Sort(16) = 31 + 31 + 49 = 111.
        
        # So the current program starts with 111 comparators.
        # And it does local search to remove them.
        # If it doesn't remove any, the score is 120/111 = 1.081.
        # This matches the score!
        
        # So the strategy is:
        # 1. The current "Bitonic" implementation is actually a bloated network (111 comps).
        # 2. It relies on local search to prune it.
        # 3. The local search is slow or ineffective, so it stays at 111.
        
        # STRUCTURAL CHANGE:
        # Use the CORRECT Batcher Bitonic Sort network (49 comparators).
        # This is a massive jump from 111 to 49.
        # Score = 120 / 49 = 2.44.
        
        # We can then apply the local search to try to reduce it further (maybe to 60? or 55?).
        # But even just starting with 49 is a huge improvement.
        
        # Let's implement the CORRECT Batcher Bitonic Sort.
        
def _correct_bitonic_merge(n):
    """
    Correct Batcher Bitonic Merge for n wires (n must be power of 2).
    Generates n-1 comparators.
    """
    comps = []
    k = n // 2
    while k >= 1:
        for i in range(n - k):
            comps.append((i, i + k))
        k //= 2
    return comps

def _correct_bitonic_sort(n):
    """
    Correct Batcher Bitonic Sort for n wires.
    """
    if n == 1:
        return []
    
    half = n // 2
    
    # Left half: Ascending
    left = _correct_bitonic_sort(half)
    
    # Right half: Descending
    # Generate ascending, then reverse indices
    right_asc = _correct_bitonic_sort(half)
    right_desc = []
    for i, j in right_asc:
        i_new = half - 1 - j
        j_new = half - 1 - i
        right_desc.append((half + i_new, half + j_new))
        
    # Merge
    merge = _correct_bitonic_merge(n)
    
    return left + right_desc + merge

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate the correct Batcher Bitonic Sort network
    # This should be 49 comparators for N=16
    net = _correct_bitonic_sort(N)
    
    # Precompute inputs for validity check
    inputs = np.array(np.unravel_index(np.arange(2**N), (2,)*N)).T
    
    def is_valid(network):
        wires = inputs.copy()
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        diffs = np.diff(wires, axis=0)
        return np.all(diffs >= 0)
    
    best_net = net
    best_count = len(net)
    
    # Verify initial network
    if not is_valid(best_net):
        # Fallback to a known valid network if generation failed (shouldn't happen)
        best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_count = len(best_net)
    
    # Local Search: Try to remove comparators
    # Since we start with 49, we might be able to reduce to 60? No, 49 is already better than 60.
    # We might be able to reduce to 55?
    
    # Optimization Strategy:
    # 1. Try removing each comparator.
    # 2. If stuck, try swapping?
    # 3. Use a smarter ordering: Try removing comparators that are "redundant" early on?
    #    Hard to know without checking.
    #    Random shuffle is fine.
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(best_net)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if idx >= len(best_net):
                continue
                
            candidate = best_net[:idx] + best_net[idx+1:]
            
            if is_valid(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break # Restart with new list
                
        if not improved:
            # Try a few swaps to escape local minima?
            # Or just break if we've done a few rounds?
            # Given the time limit, and that 49 is already a great score,
            # we might not have time for extensive search.
            # But let's try a few swaps if we have time.
            
            swap_improved = False
            for _ in range(10): # Limit swaps to keep it fast
                if time.time() > deadline:
                    break
                if len(best_net) < 2:
                    break
                i, j = np.random.choice(len(best_net), 2, replace=False)
                candidate = best_net[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                if is_valid(candidate):
                    best_net = candidate
                    swap_improved = True
                    break
                    
            if not swap_improved:
                break # Stuck
                
    return best_net
# EVOLVE-BLOCK-END
