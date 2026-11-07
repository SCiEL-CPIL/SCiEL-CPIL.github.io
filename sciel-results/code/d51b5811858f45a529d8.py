import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    best_val = float('inf')
    best_list = None
    
    # Check k=1
    # x = 210 * k^2
    # 210 < x <= 10210
    # k=1 -> 210 (invalid)
    # k=2 -> 840
    x1 = 210 * 4
    if 210 < x1 <= 10210:
        val = x1 - 210
        if val < best_val:
            best_val = val
            best_list = [x1]

    # Check k=2
    # x1 * x2 = 210 * m^2
    # We want min(max(x1, x2))
    # x1, x2 > 210
    # x1 < x2
    # x1 * x2 > 210^2 = 44100
    # 210 * m^2 > 44100 => m^2 > 210 => m >= 15
    
    # Upper bound for m:
    # x1, x2 <= 10210
    # x1 * x2 <= 10210^2
    # 210 * m^2 <= 10210^2
    # m^2 <= 10210^2 / 210
    # m_max = int(math.sqrt(10210^2 / 210))
    
    m_max = int(math.isqrt(10210**2 // 210))
    
    # We iterate m to find valid pairs
    # To minimize max, we should check smaller products first?
    # No, max(x1, x2) is roughly sqrt(P).
    # So smaller P -> smaller max.
    # So iterate m from 15 upwards.
    
    for m in range(15, m_max + 1):
        if time.time() > deadline:
            break
        P = 210 * m * m
        # Find divisors d of P such that 210 < d <= sqrt(P)
        # and P/d > 210
        # and d != P/d (distinct)
        
        # Iterate d up to sqrt(P)
        # But we only care about d > 210
        # So start d from 211
        limit = int(math.isqrt(P))
        # If limit < 211, no solution for this m
        if limit < 211:
            continue
            
        # We want to minimize max(x1, x2) = P/d.
        # This is minimized when d is as large as possible (close to sqrt(P)).
        # So iterate d downwards from limit to 211?
        # If we find a divisor, P/d is the max.
        # The first valid divisor found (largest d) gives the smallest max.
        
        found = False
        for d in range(limit, 210, -1):
            if P % d == 0:
                x2 = P // d
                if x2 > 210 and x2 != d:
                    val = x2 - 210
                    if val < best_val:
                        best_val = val
                        best_list = [d, x2]
                    found = True
                    break # Largest d gives smallest max, so no need to check smaller d for this m
        
        # Optimization: if best_val is small, and we are looking for < best_val,
        # we can stop if the theoretical min max for current m is >= best_val.
        # min max for this m is approx sqrt(P).
        # if math.isqrt(P) - 210 >= best_val:
        #    break # Even the best case for this m is not better, and larger m will be worse

    # Check k=3
    # x1 * x2 * x3 = 210 * m^2
    # x1, x2, x3 > 210
    # Distinct.
    # Minimize max.
    # This is harder to bound.
    # Given k=2 found some value, we only care if k=3 finds < best_val.
    # This means max <= 210 + best_val - 1.
    # So x1, x2, x3 in [211, 210 + best_val - 1].
    
    if best_val > 2: # Need at least 3 distinct numbers > 210, so max >= 213
        max_allowed = 210 + best_val - 1
        if max_allowed >= 213:
            nums = list(range(211, max_allowed + 1))
            n = len(nums)
            # Iterate combinations of 3
            # To be efficient, iterate max element first?
            # Or just iterate all combinations and check.
            # C(n, 3) can be large if best_val is large.
            # But if best_val is small (e.g. 20), n=20, C(20,3)=1140. Very fast.
            # If best_val is 100, n=100, C(100,3)=161700. Still okay.
            # If best_val is 500, n=500, C(500,3)=~20M. Might be slow.
            # But k=2 usually finds a good solution.
            
            for i in range(n):
                if time.time() > deadline:
                    break
                x1 = nums[i]
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    x2 = nums[j]
                    for k in range(j + 1, n):
                        if time.time() > deadline:
                            break
                        x3 = nums[k]
                        P = x1 * x2 * x3
                        if P % 210 != 0:
                            continue
                        Q = P // 210
                        s = int(math.isqrt(Q))
                        if s * s == Q:
                            val = x3 - 210
                            if val < best_val:
                                best_val = val
                                best_list = [x1, x2, x3]

    # Check k=4
    # x1 * x2 * x3 * x4 = 210 * m^2
    # If best_val is small (e.g., 10), max <= 220.
    # Range [211, 220] has 10 numbers. C(10, 4) = 210.
    # This is very fast.
    # Let's check k=4 up to max_val = 210 + best_val - 1
    
    if best_val > 3: # Need at least 4 distinct numbers > 210, so max >= 214
        max_allowed = 210 + best_val - 1
        if max_allowed >= 214:
            nums = list(range(211, max_allowed + 1))
            n = len(nums)
            for combo in combinations(nums, 4):
                if time.time() > deadline:
                    break
                P = combo[0] * combo[1] * combo[2] * combo[3]
                if P % 210 != 0:
                    continue
                Q = P // 210
                s = int(math.isqrt(Q))
                if s * s == Q:
                    val = combo[3] - 210 # combo[3] is max
                    if val < best_val:
                        best_val = val
                        best_list = list(combo)

    if best_list is None:
        return [840]
        
    return best_list
