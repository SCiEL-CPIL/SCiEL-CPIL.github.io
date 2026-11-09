import math
import bisect
import time

def construct():
    start_time = time.time()
    timeout = 55  # Safety margin for 60s limit
    
    n = 30
    # Calculate LCM of 1 to 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def generate_sums(weights_subset):
        sums = {0}
        for w in weights_subset:
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
                new_sums.add(s - w)
                new_sums.add(s)
            sums = new_sums
        return list(sums)

    # Generate sums for both halves
    # Using sets during generation prevents explosion of duplicates early on?
    # Actually, adding to a set in a loop is slower than list append.
    # But the size might stay smaller.
    # Let's try list generation first, then set deduplication.
    
    def generate_sums_list(weights_subset):
        sums = [0]
        for w in weights_subset:
            new_sums = []
            for s in sums:
                new_sums.append(s + w)
                new_sums.append(s - w)
                new_sums.append(s)
            sums = new_sums
        return sums

    sums1 = generate_sums_list(w1)
    sums2 = generate_sums_list(w2)
    
    # Deduplicate
    sums1 = list(set(sums1))
    sums2 = list(set(sums2))
    
    # Sort sums2 for binary search
    sums2.sort()
    
    best_abs_sum = float('inf')
    best_s1 = 0
    best_s2 = 0
    best_found = False
    
    for s1 in sums1:
        target = -s1
        pos = bisect.bisect_left(sums2, target)
        
        # Check neighbors
        candidates = []
        if pos > 0:
            candidates.append(sums2[pos - 1])
        if pos < len(sums2):
            candidates.append(sums2[pos])
        if pos + 1 < len(sums2):
            candidates.append(sums2[pos + 1])
            
        for s2 in candidates:
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs_sum:
                    best_abs_sum = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    best_found = True
                    if best_abs_sum == 1:
                        return reconstruct(best_s1, best_s2, w1, w2)
        
        if time.time() - start_time > timeout:
            break

    if not best_found:
        return [0] * 28 + [1, -1]
    
    return reconstruct(best_s1, best_s2, w1, w2)

def reconstruct(target_s1, target_s2, w1, w2):
    coeffs1 = backtrack(target_s1, w1, 0)
    coeffs2 = backtrack(target_s2, w2, 0)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    return coeffs1 + coeffs2

def backtrack(target, weights, idx):
    if idx == len(weights):
        if target == 0:
            return []
        else:
            return None
    
    w = weights[idx]
    
    # Try +1
    res = backtrack(target - w, weights, idx + 1)
    if res is not None:
        return [1] + res
        
    # Try 0
    res = backtrack(target, weights, idx + 1)
    if res is not None:
        return [0] + res
        
    # Try -1
    res = backtrack(target + w, weights, idx + 1)
    if res is not None:
        return [-1] + res
        
    return None
