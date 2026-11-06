import time
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    target_min = 128
    target_max = 256
    
    # Helper to calculate counts for a given set A (as a boolean array or set)
    def get_counts(A_bool):
        """
        Returns an array of counts for sums 0..256.
        count[s] = number of unordered pairs {a,b} in A such that a+b=s.
        A_bool is a numpy array of 0s and 1s of length 129.
        """
        conv = np.convolve(A_bool, A_bool, mode='full')
        sums = np.arange(257)
        even_mask = (sums % 2 == 0)
        mids = sums[even_mask] // 2
        diag_terms = np.zeros(257, dtype=np.int32)
        diag_terms[even_mask] = A_bool[mids]
        counts = (conv + diag_terms) // 2
        return counts

    # Helper to check validity
    def is_valid(counts):
        return np.all(counts[target_min:target_max+1] > 0)

    # Helper to get peak
    def get_peak(counts):
        return np.max(counts[target_min:target_max+1])

    # --- Phase 1: Construct a valid set using a structured approach ---
    # We know that if A is an arithmetic progression, the sums cover a range.
    # To minimize peak multiplicity, we want a "flat" spectrum if possible, but
    # additive bases usually have some variation.
    
    # A good heuristic is to start with a set that is "dense" enough to cover
    # but then prune.
    # However, pruning from a dense set (like all even numbers) might be slow.
    
    # Let's try a "greedy add" with a smarter score:
    # Score = - (peak_count_after) + lambda * (number_of_newly_covered_sums)
    # But we need to ensure coverage first.
    
    # Alternative: Start with a known good configuration.
    # For n=128, a set of size ~sqrt(2*128) ~ 16 is too small.
    # Actually, to cover 128 sums, with max multiplicity M, we need |A|^2/2 >= 128/M.
    # If M=2, |A|^2 >= 256 => |A| >= 16.
    # If M=3, |A|^2 >= 170 => |A| >= 14.
    # So small sets are possible in theory, but structure matters.
    
    # Let's use a randomized local search starting from a reasonable initial set.
    # Initial set: All multiples of 4? No, that leaves gaps.
    # Let's start with A = {0, 2, 4, ..., 128} (65 elements). This covers everything.
    # Peak multiplicity for A = even numbers:
    # Sum s is even. Number of pairs {a,b} even, a+b=s.
    # If s=128, a can be 0,2,...,128. 65 pairs?
    # Let's check: a+b=128. a=0,b=128; a=2,b=126... a=64,b=64.
    # Pairs: (0,128), (2,126), ..., (62,66), (64,64).
    # Count: 33 distinct pairs + 1 diag = 34?
    # Actually, a ranges 0..128 step 2.
    # If a < 64, b > 64. If a=64, b=64.
    # Number of a in [0, 128] such that a<=b and a+b=128.
    # a <= 128-a => 2a <= 128 => a <= 64.
    # a is even, so a in {0, 2, ..., 64}. There are 33 such values.
    # So peak is at least 33.
    
    # Can we do better?
    # If we remove some elements, we reduce the peak, but might create holes.
    # We need to fill holes by adding other elements that don't increase the peak elsewhere too much.
    
    # Strategy:
    # 1. Start with A = even numbers in [0, 128].
    # 2. Compute peak.
    # 3. Local Search:
    #    - Try removing an element x. If valid and peak decreases or stays same, keep it.
    #    - Try adding an element x. If peak decreases (unlikely) or allows removal of others later.
    #    - Better: Simulated Annealing on the set A.
    
    # Let's implement a fast SA.
    # State: A (boolean array 129).
    # Move: Flip a bit.
    # Cost: 
    #   - If invalid (any count in [128,256] is 0): Cost = 1000 + peak.
    #   - If valid: Cost = peak.
    
    # To make it fast, we need to update counts incrementally.
    # Flipping bit k changes counts for sums s where s-k in A or s=k+k.
    # Specifically, if we add k:
    #   For each j in A (j != k), sum s = j+k increases by 1.
    #   Sum s = 2k increases by 1.
    #   Wait, the unordered count update is tricky.
    
    # Let's stick to full convolution for simplicity first, as 129x129 is very fast.
    # np.convolve on length 129 arrays is ~microseconds.
    # We can do thousands of steps per second.
    
    # Initialize A
    A_bool = np.zeros(129, dtype=np.int32)
    # Start with all even numbers
    A_bool[0:129:2] = 1
    
    counts = get_counts(A_bool)
    current_peak = get_peak(counts)
    current_valid = is_valid(counts)
    
    # If not valid (it should be), we need to fix it.
    # Even numbers cover all even sums. Odd sums are NOT covered.
    # 128 is even, 129 is odd.
    # So A=evens is INVALID for odd sums in [128, 256].
    # We MUST have elements that allow odd sums.
    # This means A must contain at least one odd number.
    # If A has one odd number o, then o + even = odd.
    # To cover ALL odd sums, we need more.
    
    # Let's restart with a better initial guess.
    # A = {0, 1, 2, ..., 128} is valid, peak is high.
    # Let's try A = multiples of 4? No, same parity issue.
    # We need a mix.
    
    # Let's use a "greedy fill" to ensure validity first.
    # Start with empty set? No, too slow to cover 129 sums.
    # Start with a set that is likely valid.
    # How about A = [0, 1, ..., 128] but remove random elements?
    
    # Let's try a different initialization:
    # A = {0, 4, 8, ..., 128} (multiples of 4) UNION {1, 5, 9, ..., 125} (1 mod 4)?
    # This is just all numbers.
    
    # Let's try: A = even numbers UNION {odd numbers that are multiples of 4? No.}
    # Let's just start with the full set [0, 128] and prune.
    # Full set size 129.
    # Peak for full set:
    # Sum 128: a+b=128, a,b in [0,128].
    # a can be 0..128. b=128-a.
    # Unordered pairs: a <= b => a <= 64.
    # a in 0..64. 65 pairs.
    # So peak is 65.
    
    # Pruning from 129 elements down to ~20-30 elements is a good strategy.
    
    A_bool = np.ones(129, dtype=np.int32)
    counts = get_counts(A_bool)
    current_peak = get_peak(counts)
    current_valid = is_valid(counts)
    
    # Phase 2: Greedy Removal
    # Iterate: find an element x whose removal keeps validity and does not increase peak.
    # Prefer removal if peak decreases.
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        best_x = -1
        best_new_peak = current_peak
        
        # To speed up, we can check elements in a random order or sorted by "contribution".
        # Let's iterate through all 129 elements.
        for x in range(129):
            if A_bool[x] == 0:
                continue
            
            # Test removal
            A_bool[x] = 0
            counts_new = get_counts(A_bool)
            valid_new = is_valid(counts_new)
            
            if valid_new:
                peak_new = get_peak(counts_new)
                if peak_new < best_new_peak or (peak_new == best_new_peak and best_x == -1):
                    best_new_peak = peak_new
                    best_x = x
            else:
                # Restore immediately if invalid? No, we need to restore to test next x.
                pass
            
            # Restore
            A_bool[x] = 1
        
        if best_x != -1:
            A_bool[best_x] = 0
            counts = get_counts(A_bool)
            current_peak = best_new_peak
            current_valid = is_valid(counts)
            improved = True
            
    # Phase 3: Local Search (Simulated Annealing) to optimize peak
    # Now we have a valid set with a certain peak.
    # We can try to swap elements to reduce peak.
    
    if time.time() < deadline - 5:
        T = 10.0
        cool_rate = 0.95
        
        # Precompute list of indices for random access
        indices = np.arange(129)
        
        while T > 0.1 and time.time() < deadline - 5:
            # Pick a random move: flip a random bit
            x = np.random.choice(indices)
            
            old_val = A_bool[x]
            A_bool[x] = 1 - old_val
            
            counts_new = get_counts(A_bool)
            valid_new = is_valid(counts_new)
            peak_new = get_peak(counts_new)
            
            # Cost function
            if not valid_new:
                cost_new = 1000 + peak_new
            else:
                cost_new = peak_new
            
            if not current_valid:
                cost_old = 1000 + current_peak
            else:
                cost_old = current_peak
            
            delta = cost_new - cost_old
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept
                counts = counts_new
                current_peak = peak_new
                current_valid = valid_new
            else:
                # Reject
                A_bool[x] = old_val
            
            T *= cool_rate
    
    # Ensure we return a valid set. If SA left it invalid, we need to fix it.
    # But the cost function heavily penalizes invalidity, so it should be valid.
    # Just in case, if not valid, revert to last valid state? 
    # We didn't store last valid state explicitly, but the greedy phase ended valid.
    # If SA broke it, we can try to repair by adding elements.
    
    if not is_valid(counts):
        # Repair: Add elements until valid.
        # Greedy add
        while not is_valid(counts) and time.time() < deadline:
            best_x = -1
            best_gain = -1
            for x in range(129):
                if A_bool[x] == 1:
                    continue
                A_bool[x] = 1
                counts_temp = get_counts(A_bool)
                if is_valid(counts_temp):
                    gain = 1
                else:
                    # Count how many new sums covered
                    diff = counts_temp[target_min:target_max+1] - counts[target_min:target_max+1]
                    gain = np.sum(diff > 0)
                A_bool[x] = 0
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            A_bool[best_x] = 1
            counts = get_counts(A_bool)
    
    # Final check
    if not is_valid(counts):
        # Fallback to a known valid set (e.g., all evens + 1,5,9... or just full set)
        # Full set is always valid.
        A_bool = np.ones(129, dtype=np.int32)
    
    # Extract set
    A_set = [i for i in range(129) if A_bool[i] == 1]
    
    return A_set

# EVOLVE-BLOCK-END
