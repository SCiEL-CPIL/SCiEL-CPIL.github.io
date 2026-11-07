import time
from math import gcd
from functools import reduce
from collections import defaultdict, Counter

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Generate all (value, divisor_index) pairs
    events = []
    for i, d in enumerate(divisors):
        for k in range(1, max_val // d + 1):
            events.append((k * d, i))
    
    events.sort(key=lambda x: x[0])
    total_events = len(events)
    
    best_diff = float('inf')
    best_solution = None
    
    left = 0
    div_count = Counter()
    distinct_count = 0
    right = 0
    
    while right < total_events:
        if time.time() > deadline:
            break
        
        val_r, div_r = events[right]
        if div_count[div_r] == 0:
            distinct_count += 1
        div_count[div_r] += 1
        right += 1
        
        while distinct_count == n:
            window_span = events[right - 1][0] - events[left][0]
            
            if window_span < best_diff:
                # Solve assignment for this window
                window_candidates = defaultdict(list)
                for j in range(left, right):
                    v, d_idx = events[j]
                    window_candidates[d_idx].append(v)
                
                order = sorted(range(n), key=lambda i: len(window_candidates[i]))
                
                if len(window_candidates[order[0]]) > 0:
                    used = set()
                    assignment = [0] * n
                    
                    def backtrack(idx):
                        if idx == n:
                            return True
                        i = order[idx]
                        for val in window_candidates[i]:
                            if val not in used:
                                used.add(val)
                                assignment[i] = val
                                if backtrack(idx + 1):
                                    return True
                                used.remove(val)
                        return False
                    
                    if backtrack(0):
                        actual_span = max(assignment) - min(assignment)
                        if actual_span < best_diff:
                            best_diff = actual_span
                            best_solution = assignment[:]
                            if best_diff == 1:
                                return [int(x) for x in best_solution]
            
            # Shrink from left
            val_l, div_l = events[left]
            div_count[div_l] -= 1
            if div_count[div_l] == 0:
                distinct_count -= 1
            left += 1
    
    if best_solution is not None:
        return [int(x) for x in best_solution]
    
    # Fallback
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
