import time
import math
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s limit
    
    target_xor = 7  # {2, 5}
    
    # Generate numbers > 210, <= 10210, grouped by their squarefree kernel from {2,3,5,7}
    classes = {}
    for d in range(1, 211):
        temp = d
        is_valid = True
        for p in [2, 3, 5, 7]:
            if temp % (p*p) == 0:
                is_valid = False
                break
            while temp % p == 0:
                temp //= p
        if temp != 1:
            is_valid = False
        if not is_valid:
            continue
        
        min_s = 1
        while d * min_s * min_s <= 210:
            min_s += 1
        max_s = math.isqrt(10210 // d)
        
        nums = []
        for s in range(min_s, max_s + 1):
            n = d * s * s
            if 210 < n <= 10210:
                nums.append(n)
        
        if nums:
            classes[d] = nums
    
    all_candidates = []
    for d, nums in classes.items():
        for n in nums:
            all_candidates.append((n, d))
    
    all_candidates.sort()
    
    candidate_maxes = sorted(set(n for n, d in all_candidates))
    
    best_result = None
    best_max = float('inf')
    
    for max_val in candidate_maxes:
        if time.time() > deadline:
            break
        
        available = [(n, d) for n, d in all_candidates if n <= max_val]
        
        if len(available) == 0:
            continue
        
        # Try to find a subset with XOR = 7
        found = False
        for size in range(1, min(len(available), 6) + 1):
            if time.time() > deadline:
                break
            for combo in combinations(range(len(available)), size):
                xor_val = 0
                for i in combo:
                    xor_val ^= available[i][1]
                if xor_val == 7:
                    selected = [available[i][0] for i in combo]
                    if max(selected) < best_max:
                        best_max = max(selected)
                        best_result = selected
                    found = True
                    break
            if found:
                break
        
        if found:
            break
    
    if best_result is None:
        best_result = [840]
    
    result = [int(x) for x in best_result]
    return result
