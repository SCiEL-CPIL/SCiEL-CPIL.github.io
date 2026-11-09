"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

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
    
    # Precompute suffix sums for pruning: suffix_sum[i] = sum(divisors[i:])
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    # k=1: check if target is a divisor
    if target in divisors:
        return [target]
    
    # k=2: check all pairs
    for i in range(m):
        if time.time() > deadline:
            break
        for j in range(i + 1, m):
            if divisors[i] + divisors[j] == target:
                return [divisors[i], divisors[j]]
            if divisors[i] + divisors[j] < target:
                break  # divisors sorted descending, so further j won't help
    
    # k=3: check all triples with pruning
    for i in range(m):
        if time.time() > deadline:
            break
        if divisors[i] >= target:
            continue
        for j in range(i + 1, m):
            if divisors[i] + divisors[j] >= target:
                continue
            for k2 in range(j + 1, m):
                s = divisors[i] + divisors[j] + divisors[k2]
                if s == target:
                    return [divisors[i], divisors[j], divisors[k2]]
                if s < target:
                    break
    
    # k=4: meet-in-the-middle
    if time.time() < deadline:
        # Split divisors into two halves
        mid = m // 2
        left = divisors[:mid]
        right = divisors[mid:]
        
        # Generate all 2-sums from left and 2-sums from right
        left_sums = {}
        for i in range(len(left)):
            for j in range(i + 1, len(left)):
                s = left[i] + left[j]
                if s < target:
                    left_sums[s] = (i, j)
        
        right_sums = {}
        for i in range(len(right)):
            for j in range(i + 1, len(right)):
                s = right[i] + right[j]
                if s < target:
                    right_sums[s] = (i, j)
        
        # Also check 1+3 splits
        for i in range(m):
            rem = target - divisors[i]
            if rem <= 0:
                continue
            # Check if rem can be made from 3 divisors not including i
            # This is expensive, so skip for now and rely on k=5+
            pass
        
        # Check 2+2 splits
        for s, (i, j) in left_sums.items():
            rem = target - s
            if rem in right_sums:
                k2, l2 = right_sums[rem]
                # Verify all four are distinct (they are by construction since left/right are disjoint)
                return [divisors[i], divisors[j], right[k2], right[l2]]
    
    # k=5..10: recursive search with improved pruning
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
        if start + remaining_k > m:
            return False
        # Max possible: take largest remaining_k from divisors[start:]
        max_possible = suffix_sum[start] - suffix_sum[start + remaining_k]
        if max_possible < remaining_sum:
            return False
        # Min possible: take smallest remaining_k from divisors[start:]
        # The smallest remaining_k elements at or after index start are the last remaining_k
        # but we need them to be at index >= start. Since start + remaining_k <= m,
        # the smallest are divisors[m-remaining_k:] but only if m-remaining_k >= start.
        # Actually, the min sum of remaining_k distinct elements from divisors[start:] 
        # is the sum of the last remaining_k elements in divisors[start:], which are
        # divisors[m-remaining_k:] if m-remaining_k >= start, else it's more complex.
        # Since divisors is sorted descending, the smallest elements are at the end.
        # The remaining_k smallest elements available from index start onward are
        # divisors[m-remaining_k : m] if m - remaining_k >= start.
        min_start = max(start, m - remaining_k)
        min_possible = suffix_sum[min_start] - suffix_sum[m]
        if min_possible > remaining_sum:
            return False
        
        for i in range(start, m - remaining_k + 1):
            if time.time() > deadline:
                return False
            d = divisors[i]
            if d > remaining_sum:
                continue
            rem_k = remaining_k - 1
            rem_s = remaining_sum - d
            if rem_k == 0:
                if rem_s == 0:
                    chosen.append(d)
                    best_result = list(chosen)
                    chosen.pop()
                    return True
                continue
            if i + 1 + rem_k > m:
                continue
            # Prune: can we achieve rem_s with rem_k items from divisors[i+1:]?
            max_rem = suffix_sum[i + 1] - suffix_sum[i + 1 + rem_k]
            if max_rem < rem_s:
                continue
            min_rem_start = max(i + 1, m - rem_k)
            min_rem = suffix_sum[min_rem_start] - suffix_sum[m]
            if min_rem > rem_s:
                break
            
            chosen.append(d)
            if search(i + 1, rem_k, rem_s, chosen):
                return True
            chosen.pop()
    
    for k in range(5, 11):
        if time.time() > deadline:
            break
        if search(0, k, target, []):
            if best_result is not None:
                return best_result
    
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
