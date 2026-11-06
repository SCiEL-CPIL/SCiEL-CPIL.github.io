import time
from math import gcd
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    def validate(indices):
        if not indices:
            return False, 0
        if len(set(indices)) != len(indices):
            return False, 0
        if any(i < 0 or i >= n for i in indices):
            return False, 0
        s = sum(weights[i] for i in indices)
        if s >= L:
            return False, 0
        return True, s

    # ============================================================
    # DYNAMIC PROGRAMMING APPROACH
    # The LCM is manageable for exact arithmetic.
    # We perform a 0/1 knapsack to find max subset sum < L.
    # ============================================================

    # Sort items by weight for potential early termination in DP
    # Actually, for standard 0/1 knapsack, order doesn't matter for correctness
    # but processing smaller weights first might help cache behavior? 
    # No, standard 1D DP is independent of order for the max value.
    
    # We want max sum < L.
    # dp[j] = max sum achievable with exactly sum j? No, that's expensive if L is large.
    # L is huge. LCM(2..48) is approx 5.3e17. We can't allocate an array of size L.
    
    # So standard 1D DP over sum is impossible.
    
    # Alternative: Branch and Bound with strong pruning is likely necessary, 
    # but the directive asks for DP. 
    # Let's re-read: "leveraging the fact that the LCM is manageable for exact arithmetic".
    # It IS manageable for exact arithmetic (Python ints), but NOT for array allocation.
    
    # However, we can use a different DP representation or a meet-in-the-middle.
    # Meet-in-the-middle is O(2^(n/2)). n=47. 2^23.5 ~ 9 million. 
    # 9 million entries in a list is feasible.
    
    # Let's try Meet-in-the-Middle.
    # Split indices into two halves.
    # Generate all subset sums for first half.
    # Generate all subset sums for second half.
    # Sort second half sums.
    # For each sum in first half, binary search in second half for largest sum < L - sum1.
    
    half1 = list(range(n // 2))
    half2 = list(range(n // 2, n))
    
    n1 = len(half1)
    n2 = len(half2)
    
    # Generate all subset sums for half1
    # n1 = 23. 2^23 = 8,388,608.
    # n2 = 24. 2^24 = 16,777,216.
    # Total ~ 25 million integers. 
    # Memory: 25 million * 8 bytes (if using array('q') or list of ints) ~ 200MB. 
    # Python list of ints is heavier (28 bytes each). 25M * 28 = 700MB. 
    # This might be too much memory or slow to generate.
    
    # Let's optimize: 
    # Instead of storing all sums, we can just sort the generated sums of the smaller half 
    # and iterate the other half.
    
    # Actually, we don't need to store the indices for the binary search part, 
    # just the sums. But we DO need the indices to return the result.
    
    # Strategy:
    # 1. Generate all (sum, mask) for half1.
    # 2. Sort by sum.
    # 3. Generate all (sum, mask) for half2.
    # 4. For each (s2, m2) in half2, find best (s1, m1) in half1 such that s1+s2 < L.
    #    We can use binary search on the sorted half1 sums.
    
    # To save memory, we can store half1 sums in a separate array and masks in another?
    # Or just a list of tuples.
    
    # Let's try to be memory efficient.
    # n1 = 23, n2 = 24.
    # Let's store half1: list of (sum, mask).
    # 8M tuples. Each tuple is a bit heavy.
    
    # Alternative: Store just sums in a sorted list, and masks in a corresponding list?
    # Or, since we only need the MAX sum, we can prune dominated entries in half1?
    # If we have two entries (s_a, m_a) and (s_b, m_b) with s_a <= s_b, 
    # s_b is always better unless m_b is invalid for some reason? 
    # No, mask doesn't matter for the sum value, only for the final reconstruction.
    # So for a given sum, we only need one mask.
    # But sums are likely unique? No, different subsets can have same sum.
    # If sums are same, any mask works. So we can keep unique sums?
    # But we need the mask.
    
    # Let's just generate the lists.
    
    def gen_subsets(items):
        """Yields (sum, mask) for all subsets of items."""
        res = []
        for mask in range(1 << len(items)):
            s = 0
            for i in range(len(items)):
                if mask & (1 << i):
                    s += weights[items[i]]
            if s < L: # Only keep valid sums
                res.append((s, mask))
        return res
    
    # This nested loop in Python might be slow for 8M items.
    # 8M * 23 operations ~ 190M ops. Might take 20-30s in Python.
    # Let's try to optimize generation.
    
    # Faster generation using iterative doubling
    def gen_subsets_fast(items):
        sums = [0]
        masks = [0]
        for i in range(len(items)):
            w = weights[items[i]]
            bit = 1 << i
            # Add new subsets that include item i
            new_sums = [s + w for s in sums if s + w < L]
            new_masks = [m | bit for m, s in zip(masks, sums) if s + w < L]
            sums.extend(new_sums)
            masks.extend(new_masks)
        # We need to pair them back up
        # The above logic is slightly flawed because masks and sums grow together.
        # Let's do it carefully.
        
        pairs = [(0, 0)]
        for i in range(len(items)):
            w = weights[items[i]]
            bit = 1 << i
            current_len = len(pairs)
            for j in range(current_len):
                s, m = pairs[j]
                ns = s + w
                if ns < L:
                    pairs.append((ns, m | bit))
        return pairs

    # This is still O(2^n * n) if we do it naively? No, it's O(2^n) total appends.
    # Each step doubles the list (roughly).
    # Total operations ~ sum(2^i) ~ 2^(n+1).
    # For n=23, ~ 16M appends. This should be fast enough in Python (maybe 5-10s).
    
    # Let's do this.
    
    if check_time():
        # Fallback
        return [0]

    print(f"Generating half 1 subsets...", file=sys.stderr)
    pairs1 = gen_subsets_fast(half1)
    print(f"Generated {len(pairs1)} subsets for half 1", file=sys.stderr)
    
    if check_time():
        # Sort and find best single half?
        if pairs1:
            pairs1.sort(key=lambda x: x[0])
            best_s, best_m = pairs1[-1]
            # Convert mask to indices
            indices = []
            for i in range(n1):
                if best_m & (1 << i):
                    indices.append(half1[i])
            return indices
        return [0]

    print(f"Generating half 2 subsets...", file=sys.stderr)
    pairs2 = gen_subsets_fast(half2)
    print(f"Generated {len(pairs2)} subsets for half 2", file=sys.stderr)
    
    if check_time():
        if pairs2:
            pairs2.sort(key=lambda x: x[0])
            best_s, best_m = pairs2[-1]
            indices = []
            for i in range(n2):
                if best_m & (1 << i):
                    indices.append(half2[i])
            return indices
        return [0]

    # Sort pairs1 by sum
    pairs1.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums1 = [p[0] for p in pairs1]
    
    import bisect
    
    best_total_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate over pairs2
    # For each (s2, m2) in pairs2, we want max s1 in sums1 such that s1 + s2 < L
    # i.e., s1 < L - s2
    
    for s2, m2 in pairs2:
        if check_time():
            break
            
        limit = L - s2
        if limit <= 0:
            continue
            
        # Find rightmost index in sums1 where value < limit
        # bisect_left gives first index where value >= limit
        # So we want index - 1
        pos = bisect.bisect_left(sums1, limit)
        
        if pos > 0:
            idx1 = pos - 1
            s1 = sums1[idx1]
            total = s1 + s2
            if total > best_total_sum:
                best_total_sum = total
                best_mask1 = pairs1[idx1][1]
                best_mask2 = m2
    
    if check_time():
        if best_total_sum > -1:
            indices = []
            for i in range(n1):
                if best_mask1 & (1 << i):
                    indices.append(half1[i])
            for i in range(n2):
                if best_mask2 & (1 << i):
                    indices.append(half2[i])
            return indices
        # Fallback to greedy
        s = 0
        chosen = []
        for i in range(n):
            if s + weights[i] < L:
                s += weights[i]
                chosen.append(i)
        return chosen

    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(half1[i])
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(half2[i])
    
    # Final validation
    valid, s = validate(indices)
    if valid:
        return indices
    
    # Fallback
    s = 0
    chosen = []
    for i in range(n):
        if s + weights[i] < L:
            s += weights[i]
            chosen.append(i)
    return chosen
