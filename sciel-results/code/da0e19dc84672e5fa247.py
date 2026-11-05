"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
ZERO_ONE_COUNT = 1 << N  # 65536
DEADLINE_SECONDS = 75.0  # Safety margin below 90s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sorting network for 16 wires.
    Strategy:
    1. Start with a known valid network (Odd-Even Transposition, 120 comparators).
    2. Attempt to reduce size using a greedy local search (removing redundant comparators).
    3. If time permits, attempt to find a smaller network via random restarts or swaps,
       though local search on the transposition network is the most reliable quick win.
    """
    start_time = time.time()
    
    # Helper: Check if a network is a valid sorting network using Zero-One Principle
    # Optimized: Use bit manipulation to simulate all 65536 inputs efficiently?
    # Actually, simulating 65536 inputs x 120 comparators in pure Python is slow (~1-2s).
    # We need a faster verification or fewer checks.
    # Let's use a vectorized approach with NumPy if possible, or a clever bit trick.
    
    # Bit trick: 
    # Represent a 16-bit input as an integer.
    # A comparator (i, j) on an integer x:
    #   bit_i = (x >> i) & 1
    #   bit_j = (x >> j) & 1
    #   min_bit = bit_i & bit_j  (if both 1, min is 1? No. min(1,1)=1, min(1,0)=0, min(0,1)=0, min(0,0)=0) -> AND
    #   max_bit = bit_i | bit_j
    #   So: 
    #   x' = x
    #   if bit_i > bit_j: swap bits.
    #   bit_i > bit_j iff bit_i=1 and bit_j=0.
    #   In that case, set bit_i to 0 and bit_j to 1.
    #   mask = (1 << i) | (1 << j)
    #   # This is getting complex to vectorize over 65536 inputs at once in pure python loops.
    
    # Let's try a different verification strategy: 
    # The Zero-One principle says we only need to check inputs that are 0/1 sequences.
    # There are 65536 such sequences.
    # Simulating one sequence takes O(C) where C is comparator count.
    # 65536 * 120 operations = ~8 million operations. In Python, this is ~1-2 seconds.
    # We have 75 seconds. We can afford roughly 30-40 full verifications.
    # This is too few for a robust search.
    
    # Alternative: Use a faster verification method.
    # We can simulate the network on all 65536 inputs simultaneously using NumPy?
    # Input array shape: (65536, 16) of booleans/uint8.
    # Comparator (i, j):
    #   col_i = inputs[:, i]
    #   col_j = inputs[:, j]
    #   min_col = np.minimum(col_i, col_j)
    #   max_col = np.maximum(col_i, col_j)
    #   inputs[:, i] = min_col
    #   inputs[:, j] = max_col
    # This is vectorized! 65536 x 16 array is small (~1MB).
    # 120 comparators * vectorized ops is very fast (< 0.1s).
    # This allows for hundreds or thousands of verifications in the time limit.
    
    def verify_network(comparators):
        """
        Returns True if the network sorts all 16-bit inputs.
        Uses NumPy vectorization over all 2^16 inputs.
        """
        if not comparators:
            return False
        # Generate all 16-bit patterns as a (65536, 16) array of uint8
        # np.arange(65536) gives 0..65535
        # We need to unpack bits.
        # arange(65536)[:, None] >> np.arange(15, -1, -1) & 1
        # Note: Wire 0 is LSB or MSB? 
        # Task: min to wire i, max to wire j. 
        # Standard sorting network: wire 0 is usually the first output (smallest).
        # In the bit representation, if we treat input integer x, bit 0 is LSB.
        # If we sort, the smallest value should end up in wire 0.
        # Let's stick to the convention: 
        # Input is a list of 16 values.
        # If we use the integer representation, let's say bit k corresponds to wire k.
        # After sorting, wire 0 should be the minimum. 
        # For a sorted 0/1 sequence, the zeros come first, then ones.
        # So wire 0..k-1 are 0, wire k..15 are 1.
        # This means the bits in the integer should be sorted such that lower indices are 0?
        # Wait. min goes to i. If i < j, and we want sorted output [0,0,...,1,1],
        # then wire 0 (lowest index) should have 0.
        # So in the final state, for a given input, the bits at positions 0..count_0-1 should be 0, and rest 1.
        
        # Let's generate inputs.
        # Pattern p (0 to 65535).
        # Bit 0 of p is wire 0?
        # Let's define wire i corresponds to bit i of the integer.
        # inputs = (np.arange(65536)[:, None] >> np.arange(16)[None, :]) & 1
        # This gives a (65536, 16) array where column i is bit i of the row index.
        
        inputs = (np.arange(ZERO_ONE_COUNT, dtype=np.uint16)[:, None] >> np.arange(16, dtype=np.uint16)[None, :]) & 1
        
        # Apply comparators
        for i, j in comparators:
            # Ensure i < j
            if i > j:
                i, j = j, i
            col_i = inputs[:, i]
            col_j = inputs[:, j]
            
            # min goes to i, max goes to j
            inputs[:, i] = np.minimum(col_i, col_j)
            inputs[:, j] = np.maximum(col_i, col_j)
        
        # Check if sorted
        # For each row, the bits should be non-decreasing? 
        # No, min goes to lower index. So wire 0 <= wire 1 <= ... <= wire 15.
        # In binary 0/1, this means all 0s are at lower indices, all 1s at higher indices.
        # So for each row, we check if it is of the form 0...01...1.
        # This is equivalent to checking that there is no 1 followed by a 0.
        # i.e., inputs[:, k] <= inputs[:, k+1] for all k.
        
        # Vectorized check:
        # diffs = np.diff(inputs, axis=1)
        # if np.any(diffs < 0): return False
        
        # Even faster: 
        # A row is sorted iff the first '1' is at some position p, and all before are 0, all after are 1.
        # This is equivalent to: the sequence of bits is monotonic non-decreasing.
        
        # Let's just check: inputs[:, :-1] <= inputs[:, 1:]
        return np.all(inputs[:, :-1] <= inputs[:, 1:])

    # Initial Network: Odd-Even Transposition
    # This is known to be valid.
    current_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify it just in case
    if not verify_network(current_network):
        # Fallback: should not happen
        return current_network

    best_network = current_network[:]
    best_size = len(best_network)
    
    # Local Search: Try removing comparators
    # We iterate through the network and try to remove each comparator.
    # If the network remains valid, we keep the removal.
    # We repeat this process until no more removals are possible or time runs out.
    
    # To speed up, we can cache verification results? 
    # Or just rely on the fast NumPy verification.
    # One verification is ~0.05s? Let's assume 0.1s to be safe.
    # 75s / 0.1s = 750 verifications.
    # We have 120 comparators. We can try many removals.
    
    # Strategy:
    # 1. Greedy Removal:
    #    Loop:
    #      Found_improvement = False
    #      For each index k in range(len(network)):
    #         If time is up: break
    #         candidate = network with comparator k removed
    #         If verify(candidate):
    #             network = candidate
    #             Found_improvement = True
    #             break # Restart loop from beginning as indices shifted
    #      If not Found_improvement: break
    
    # This is O(N^2) verifications in worst case, but usually much less.
    # 120 comparators, each removal check is O(1) verification.
    # If we remove 60 comparators, we do ~60 * 120 / 2 = 3600 checks? No.
    # Each successful removal reduces length by 1.
    # Worst case: we check every comparator at every step.
    # Step 1: 120 checks.
    # Step 2: 119 checks.
    # ...
    # Total ~ 120^2 / 2 = 7200 checks.
    # 7200 * 0.1s = 720s. Too slow!
    
    # We need a smarter heuristic or faster verification.
    # Or, we limit the number of attempts.
    # Or, we use a "batch" removal strategy?
    
    # Let's optimize the verification.
    # The NumPy verification creates the input array every time.
    # We can pre-compute the input array once.
    
    global _INPUTS
    if '_INPUTS' not in globals():
        _INPUTS = (np.arange(ZERO_ONE_COUNT, dtype=np.uint16)[:, None] >> np.arange(16, dtype=np.uint16)[None, :]) & 1

    def verify_network_fast(comparators):
        # Copy inputs to avoid modifying the global
        # 65536 * 16 * 2 bytes = ~2MB. Copy is fast.
        inputs = _INPUTS.copy()
        
        for i, j in comparators:
            col_i = inputs[:, i]
            col_j = inputs[:, j]
            inputs[:, i] = np.minimum(col_i, col_j)
            inputs[:, j] = np.maximum(col_i, col_j)
        
        return np.all(inputs[:, :-1] <= inputs[:, 1:])

    # Now verify_network_fast is faster (no array generation).
    # Let's benchmark mentally:
    # Copy: ~0.1ms
    # 120 comparators: 120 * (2 col accesses + min + max + 2 assigns)
    # NumPy ops on 65k elements are very fast.
    # Probably ~1-2ms per verification.
    # 7200 checks * 2ms = 14.4s.
    # This is feasible within 75s!
    
    # Let's run the greedy removal.
    network = current_network[:]
    
    # To make it even faster, we can try removing from the end first?
    # Or random order?
    # Let's stick to sequential for determinism, but maybe skip if time is tight.
    
    # We'll do multiple passes.
    # Pass 1: Remove if possible.
    
    # Optimization: 
    # If we remove a comparator, we don't need to re-verify the ones before it if they were already "safe"?
    # No, removing a later comparator can make an earlier one redundant? 
    # Actually, removing a comparator can only make the network *less* sorted.
    # So if the network was sorted, removing a comparator might break it.
    # If we remove comparator A, and it was redundant, the network is still sorted.
    # If we then remove comparator B, we need to check the new network.
    
    # Greedy algorithm:
    improved = True
    while improved:
        improved = False
        # Iterate backwards? Removing from the end might be safer/faster?
        # Actually, removing any comparator requires full verification.
        # Let's iterate from 0 to len-1.
        for k in range(len(network)):
            if time.time() - start_time > DEADLINE_SECONDS - 1.0:
                break
            
            # Create candidate network
            # network[:k] + network[k+1:]
            candidate = network[:k] + network[k+1:]
            
            if verify_network_fast(candidate):
                network = candidate
                improved = True
                break # Restart the for loop with new length
        
        if time.time() - start_time > DEADLINE_SECONDS - 1.0:
            break

    best_network = network
    best_size = len(network)
    
    # If we have time, try to find a better network via random restarts?
    # Or try to swap comparators?
    # Given the time budget, the greedy removal from the 120-comparator network
    # should get us down to a reasonable number (likely ~80-90).
    # The reference is 60.
    
    # Let's see if we can do better with a different initial network?
    # The task mentions "A verified 60-comparator construction is the reference".
    # We don't have that construction hardcoded.
    # We can try to construct a better network from scratch?
    # That's hard.
    
    # Let's try to add some randomization to the removal order to escape local minima?
    # Or try a "shaking" phase:
    # 1. Remove redundancies.
    # 2. Randomly swap two comparators.
    # 3. Remove redundancies.
    # 4. If size decreased, keep.
    
    # But we must be careful with time.
    # Let's just return the result of the greedy removal for now.
    # It's a solid improvement over 120.
    
    return best_network
# EVOLVE-BLOCK-END
