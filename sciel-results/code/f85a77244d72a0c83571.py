# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisor_set = set(divisors)
    divisors.sort()
    n_divs = len(divisors)
    
    # k=1: check if target itself is a divisor
    if target in divisor_set:
        return [target]
    
    # k=2: O(n) using set
    for d in divisors:
        complement = target - d
        if complement in divisor_set and complement != d and complement > 0:
            return [d, complement]
    
    # k=3: O(n^2) using set
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            complement = target - a - b
            if complement in divisor_set and complement > b:
                return [a, b, complement]
    
    # k=4: Meet-in-the-middle
    # Generate all pair sums with their indices
    pair_sums = {}  # sum -> list of (i, j) pairs
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            s = a + b
            if s >= target:
                break
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
    
    # Now look for two disjoint pairs that sum to target
    for s1 in pair_sums:
        s2 = target - s1
        if s2 not in pair_sums:
            continue
        if s2 < s1:  # avoid duplicates
            continue
        for (i1, j1) in pair_sums[s1]:
            for (i2, j2) in pair_sums[s2]:
                # Check disjointness
                if len({i1, j1, i2, j2}) == 4:
                    return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
    
    # k=5 and above: Backtracking with pruning
    # Precompute suffix sums for pruning
    suffix_sums = [0] * (n_divs + 1)
    for i in range(n_divs - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + divisors[i]
    
    def backtrack(start_idx, remaining_k, remaining_sum, chosen):
        if time.time() > deadline:
            return None
        
        if remaining_k == 0:
            if remaining_sum == 0:
                return list(chosen)
            return None
        
        # Not enough elements left
        if n_divs - start_idx < remaining_k:
            return None
        
        # Minimum possible sum: pick the smallest remaining_k from start_idx
        min_possible = sum(divisors[start_idx:start_idx + remaining_k])
        if remaining_sum < min_possible:
            return None
        
        # Maximum possible sum: pick the largest remaining_k from start_idx onwards
        # The largest remaining_k elements in divisors[start_idx:] are the last remaining_k of the whole array
        # but only if start_idx <= n_divs - remaining_k
        if start_idx <= n_divs - remaining_k:
            max_possible = suffix_sums[n_divs - remaining_k] - suffix_sums[n_divs] + sum(divisors[n_divs - remaining_k:])
            # Actually, simpler: the max sum of remaining_k elements from index start_idx onward
            # is the sum of the last remaining_k elements of divisors (since they're sorted)
            max_possible = sum(divisors[n_divs - remaining_k:])
        else:
            max_possible = suffix_sums[start_idx]
        
        if remaining_sum > max_possible:
            return None
        
        # Try picking each divisor from start_idx
        for i in range(start_idx, n_divs):
            d = divisors[i]
            if d > remaining_sum:
                break
            
            # Pruning: even if we pick d, can we still achieve remaining_sum - d with remaining_k - 1 more?
            new_remaining = remaining_sum - d
            new_k = remaining_k - 1
            
            if new_k == 0:
                if new_remaining == 0:
                    chosen.append(d)
                    result = list(chosen)
                    chosen.pop()
                    return result
                continue
            
            # Check if it's possible to get new_remaining with new_k elements from i+1 onward
            min_next = sum(divisors[i + 1:i + 1 + new_k]) if i + 1 + new_k <= n_divs else None
            if min_next is None:
                continue
            if new_remaining < min_next:
                continue
            
            max_next = sum(divisors[n_divs - new_k:]) if i + 1 <= n_divs - new_k else suffix_sums[i + 1]
            if new_remaining > max_next:
                continue
            
            chosen.append(d)
            result = backtrack(i + 1, new_k, new_remaining, chosen)
            if result is not None:
                return result
            chosen.pop()
        
        return None
    
    for k in range(5, min(257, n_divs + 1)):
        result = backtrack(0, k, target, [])
        if result is not None:
            return result
    
    # Fallback: greedy (should not reach here)
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
    
    return chosen
# EVOLVE-BLOCK-END
