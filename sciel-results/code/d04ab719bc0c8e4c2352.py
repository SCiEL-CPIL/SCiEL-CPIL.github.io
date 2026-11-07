import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    best_val = float('inf')
    best_list = None
    
    # Check k=1: x = 210 * t^2, 210 < x <= 10210
    for t in range(2, 100):
        x = 210 * t * t
        if x > 10210:
            break
        if 210 < x:
            val = x - 210
            if val < best_val:
                best_val = val
                best_list = [x]
    
    # Exhaustive search for k>=2 with max <= 230 (to beat score 21)
    # Numbers: 211 to 230 (20 numbers)
    search_range = list(range(211, 231))
    
    for size in range(2, 11):
        if time.time() > deadline:
            break
        for combo in combinations(search_range, size):
            P = 1
            for x in combo:
                P *= x
            val = 210 * P
            s = math.isqrt(val)
            if s * s == val:
                current_max = max(combo) - 210
                if current_max < best_val:
                    best_val = current_max
                    best_list = list(combo)
    
    # If best_val > 20, search for pairs/triples with larger max
    if best_val > 20:
        max_search = min(10210, 210 + best_val - 1)
        
        # Search pairs
        for i in range(211, max_search + 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, max_search + 1):
                P = i * j
                val = 210 * P
                s = math.isqrt(val)
                if s * s == val:
                    current_max = j - 210
                    if current_max < best_val:
                        best_val = current_max
                        best_list = [i, j]
        
        # Search triples (limited range for feasibility)
        max_trip = min(max_search, 300)
        if max_trip >= 231:
            for i in range(211, max_trip + 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, max_trip + 1):
                    for k in range(j + 1, max_trip + 1):
                        P = i * j * k
                        val = 210 * P
                        s = math.isqrt(val)
                        if s * s == val:
                            current_max = k - 210
                            if current_max < best_val:
                                best_val = current_max
                                best_list = [i, j, k]
    
    if best_list is None:
        return [840]
    
    return [int(x) for x in best_list]
