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
    
    # Precompute suffix sums for pruning: suffix_sum[i] = sum(divisors[i:])
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    best_result = None
    
    # Try increasing number of terms: 1, 2, 3, ...
    for k in range(1, min(m, 256) + 1):
        if time.time() > deadline:
            break
        
        if k == 1:
            # k=1: just check if target is a divisor
            if target in divisors:
                return [target]
            continue
        
        if k <= 8:
            # Recursive search with pruning
            def search(start, remaining_k, remaining_sum, chosen):
                nonlocal best_result
                if time.time() > deadline:
                    return False
                if remaining_k == 0:
                    if remaining_sum == 0:
                        best_result = list(chosen)
                        return True
                    return False
                # Not enough divisors left
                if start + remaining_k > m:
                    return False
                # Max possible sum: take the largest remaining_k from divisors[start:]
                max_possible = suffix_sum[start] - suffix_sum[start + remaining_k]
                if max_possible < remaining_sum:
                    return False
                # Min possible sum: take the smallest remaining_k from divisors[start:]
                # These are the last remaining_k elements in the array
                # But they must be at index >= start, so we need m - remaining_k >= start
                # Which is guaranteed by start + remaining_k <= m
                min_possible = suffix_sum[m - remaining_k]
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
                    # Check if we can achieve rem_s with rem_k items from divisors[i+1:]
                    max_rem = suffix_sum[i + 1] - suffix_sum[i + 1 + rem_k]
                    if max_rem < rem_s:
                        continue
                    min_rem = suffix_sum[m - rem_k]
                    if min_rem > rem_s:
                        break
                    
                    chosen.append(d)
                    if search(i + 1, rem_k, rem_s, chosen):
                        return True
                    chosen.pop()
            
            if search(0, k, target, []):
                if best_result is not None:
                    return best_result
        else:
            # For k > 8, break to fallback
            break
    
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
