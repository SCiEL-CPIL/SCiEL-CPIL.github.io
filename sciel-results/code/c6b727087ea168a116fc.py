import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    L = 1
    for d in divisors:
        L = L * d // math.gcd(L, d)
    
    start_time = time.time()
    time_limit = 50.0
    
    # Iterate R (span) from 1 upwards
    # Max span is L-1 (since we can always pick multiples within one LCM period)
    # Actually, max span is bounded by L.
    
    for R in range(1, L + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Check if there exists a starting position s in [0, L-1]
        # such that for all i, [s, s+R] contains a multiple of divisors[i]
        # and distinct multiples can be assigned.
        
        # 1. Filter s using necessary condition: each divisor has at least one multiple in [s, s+R]
        valid_s = bytearray([1]) * L
        
        for d in divisors:
            good_mask = bytearray([0]) * L
            # Multiples m = k*d
            # s is good if exists k such that s <= k*d <= s+R
            # => k*d - R <= s <= k*d
            # We need to mark s in [0, L-1]
            
            # Range of k:
            # min s = 0 => k*d >= 0 => k >= 0
            # max s = L-1 => k*d <= L-1 + R => k <= (L-1+R)/d
            k_min = 0
            k_max = (L - 1 + R) // d
            
            for k in range(k_min, k_max + 1):
                m = k * d
                s_start = max(0, m - R)
                s_end = min(L - 1, m)
                if s_start <= s_end:
                    good_mask[s_start:s_end+1] = b'\x01' * (s_end - s_start + 1)
            
            # Intersect
            for i in range(L):
                if not good_mask[i]:
                    valid_s[i] = 0
            
            # Early exit if no valid s
            if not any(valid_s):
                break
        
        if not any(valid_s):
            continue
            
        # 2. For each valid s, try to find a distinct assignment
        # We can iterate through s, but to be faster, we can pick a few or all.
        # Since R is small, the number of valid s might be small or the DFS will be fast.
        
        for s in range(L):
            if not valid_s[s]:
                continue
                
            if time.time() - start_time > time_limit:
                break
                
            # Generate candidates for this s and R
            candidates = []
            valid = True
            for d in divisors:
                first = ((s + d - 1) // d) * d
                if first == 0:
                    first = d
                cand_list = []
                m = first
                while m <= s + R:
                    if m > 0:
                        cand_list.append(m)
                    m += d
                if not cand_list:
                    valid = False
                    break
                candidates.append(cand_list)
            
            if not valid:
                continue
                
            # Sort indices by number of candidates (MRV)
            indices = list(range(len(divisors)))
            indices.sort(key=lambda i: len(candidates[i]))
            
            used = set()
            solution = [None] * len(divisors)
            
            # DFS
            found = False
            def dfs(idx):
                nonlocal found
                if found:
                    return
                if idx == len(indices):
                    found = True
                    return
                i = indices[idx]
                for val in candidates[i]:
                    if val in used:
                        continue
                    used.add(val)
                    solution[i] = val
                    dfs(idx + 1)
                    if found:
                        return
                    used.remove(val)
                    solution[i] = None
            
            dfs(0)
            
            if found:
                return [int(x) for x in solution]
                
    # Fallback
    return divisors[:]

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    L = 1
    for d in divisors:
        L = L * d // math.gcd(L, d)
    
    start_time = time.time()
    time_limit = 50.0
    
    for R in range(1, L + 1):
        if time.time() - start_time > time_limit:
            break
            
        valid_s = bytearray([1]) * L
        
        for d in divisors:
            good_mask = bytearray([0]) * L
            
            k_min = 0
            k_max = (L - 1 + R) // d
            
            for k in range(k_min, k_max + 1):
                m = k * d
                s_start = max(0, m - R)
                s_end = min(L - 1, m)
                if s_start <= s_end:
                    good_mask[s_start:s_end+1] = b'\x01' * (s_end - s_start + 1)
            
            for i in range(L):
                if not good_mask[i]:
                    valid_s[i] = 0
            
            if not any(valid_s):
                break
        
        if not any(valid_s):
            continue
            
        for s in range(L):
            if not valid_s[s]:
                continue
                
            if time.time() - start_time > time_limit:
                break
                
            candidates = []
            valid = True
            for d in divisors:
                first = ((s + d - 1) // d) * d
                if first == 0:
                    first = d
                cand_list = []
                m = first
                while m <= s + R:
                    if m > 0:
                        cand_list.append(m)
                    m += d
                if not cand_list:
                    valid = False
                    break
                candidates.append(cand_list)
            
            if not valid:
                continue
                
            indices = list(range(len(divisors)))
            indices.sort(key=lambda i: len(candidates[i]))
            
            used = set()
            solution = [None] * len(divisors)
            
            found = False
            def dfs(idx):
                nonlocal found
                if found:
                    return
                if idx == len(indices):
                    found = True
                    return
                i = indices[idx]
                for val in candidates[i]:
                    if val in used:
                        continue
                    used.add(val)
                    solution[i] = val
                    dfs(idx + 1)
                    if found:
                        return
                    used.remove(val)
                    solution[i] = None
            
            dfs(0)
            
            if found:
                return [int(x) for x in solution]
                
    return divisors[:]
# EVOLVE-BLOCK-END
