"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute divisors of 5040
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    # Filter: only divisors that could help (<= target)
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute suffix sums: suffix_sum[i] = sum(divisors[i:])
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # Precompute prefix sums for quick max-k-sum lookups
    # prefix_sum[i] = sum(divisors[0:i])
    prefix_sum = [0] * (m + 1)
    for i in range(1, m + 1):
        prefix_sum[i] = prefix_sum[i - 1] + divisors[i - 1]
    
    # Max sum of k items starting at index s: sum(divisors[s:s+k]) = prefix_sum[s+k] - prefix_sum[s]
    # Min sum of k items from indices s..m-1: sum of last k items = suffix_sum[m-k]
    # (valid when m-k >= s, i.e., k <= m-s)
    
    best_result = None
    
    def search(start, remaining_k, remaining_sum, chosen):
        nonlocal best_result
        if time.time() > deadline:
            return False
        if remaining_k == 0:
            if remaining_sum == 0:
                best_result = list(chosen)
                return True
            return False
        if remaining_k == 1:
            # Binary search or linear scan for exact value
            # Need to find divisors[i] == remaining_sum for some i >= start
            # Since divisors is sorted descending, we can scan
            for i in range(start, m):
                if divisors[i] == remaining_sum:
                    chosen.append(divisors[i])
                    best_result = list(chosen)
                    chosen.pop()
                    return True
                if divisors[i] < remaining_sum:
                    break
            return False
        
        # Not enough divisors left
        if start + remaining_k > m:
            return False
        
        # Max possible sum: take the largest remaining_k from start
        max_possible = prefix_sum[start + remaining_k] - prefix_sum[start]
        if max_possible < remaining_sum:
            return False
        
        # Min possible sum: take the smallest remaining_k (last remaining_k in array)
        # Valid since start <= m - remaining_k (checked above)
        min_possible = suffix_sum[m - remaining_k]
        if min_possible > remaining_sum:
            return False
        
        rem_k = remaining_k - 1
        # For each candidate d at index i:
        #   rem_s = remaining_sum - d
        #   Need: min_rem <= rem_s <= max_rem where
        #   max_rem = sum of largest rem_k from i+1..m-1 = prefix_sum[i+1+rem_k] - prefix_sum[i+1]
        #   min_rem = sum of smallest rem_k = suffix_sum[m-rem_k]
        # Also: d <= remaining_sum (obvious)
        # And: i + 1 + rem_k <= m => i <= m - rem_k - 1 = m - remaining_k
        
        upper_i = m - remaining_k  # inclusive upper bound for i
        for i in range(start, upper_i + 1):
            d = divisors[i]
            if d > remaining_sum:
                continue
            
            rem_s = remaining_sum - d
            
            # Max from remaining: largest rem_k after i
            max_rem = prefix_sum[i + 1 + rem_k] - prefix_sum[i + 1]
            if max_rem < rem_s:
                # d too small, rem_s too big; for smaller d (later), rem_s gets even bigger
                # But max_rem also decreases as i increases... so we can't simply break
                # Actually as i increases, d decreases, rem_s increases, and max_rem decreases
                # So this condition will keep getting worse. We can break.
                break
            
            # Min from remaining: smallest rem_k overall
            min_rem = suffix_sum[m - rem_k]
            if min_rem > rem_s:
                # d too big, rem_s too small; for smaller d (later), rem_s increases
                # min_rem is constant, so eventually rem_s >= min_rem
                # We can't break here; we continue to smaller d
                continue
            
            chosen.append(d)
            if search(i + 1, rem_k, rem_s, chosen):
                return True
            chosen.pop()
        
        return False
    
    # Try increasing number of terms
    for k in range(1, min(m, 256) + 1):
        if time.time() > deadline:
            break
        
        if k == 1:
            # Check if target is a divisor
            for d in divisors:
                if d == target:
                    return [target]
                if d < target:
                    break
            continue
        
        best_result = None
        if search(0, k, target, []):
            if best_result is not None:
                return best_result
        else:
            # If search exhausted without finding solution, try next k
            continue
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    assert remainder == 0, f"Greedy failed, remainder={remainder}"
    return chosen
# EVOLVE-BLOCK-END
