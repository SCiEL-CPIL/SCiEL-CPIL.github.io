import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin within 60s limit
    
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = reduce(lcm, divisors)  # 30030
    
    # Generate all multiples for each divisor in [1, period-1]
    # We duplicate them shifted by period to handle wrap-around windows
    events = []
    for i, d in enumerate(divisors):
        # k goes from 1 to period//d - 1
        # If period//d is 1, there are no multiples in (0, period)
        for k in range(1, period // d):
            m = k * d
            events.append((m, i))
            events.append((m + period, i))
            
    if not events:
        # Fallback if something went wrong, though period is large enough
        return [int(d) for d in divisors]

    # Sort events by value
    events.sort(key=lambda x: x[0])
    
    # Sliding window to find minimal span covering all divisors
    # We want a window [L, R] such that all n divisors are present
    # and R - L is minimized.
    
    best_span = float('inf')
    best_L = 0
    best_R = 0
    best_window_indices = None # Indices in events list for the best window
    
    # We will store the top few best windows to handle distinctness conflicts
    # But likely the first one works.
    candidate_windows = []
    
    left = 0
    count = [0] * n
    unique_divs = 0
    
    # We only need to look for windows of size < period
    # If the window size exceeds period, it's not optimal compared to a wrapped one
    # But our events are sorted, so we just slide.
    
    for right in range(len(events)):
        val, div_idx = events[right]
        
        # Expand window to include events[right]
        if count[div_idx] == 0:
            unique_divs += 1
        count[div_idx] += 1
        
        # Shrink window from left while all divisors are covered
        while unique_divs == n:
            current_span = events[right][0] - events[left][0]
            
            # We only care about spans smaller than period
            if current_span < period and current_span < best_span:
                best_span = current_span
                best_L = events[left][0]
                best_R = events[right][0]
                
                # Save this window's events for reconstruction
                # We need to know which events are in [best_L, best_R]
                # Since we are in the loop, we can slice events[left:right+1]
                # But we need to be careful with duplicates in events list
                # events is a list of (val, idx).
                # We will store the slice indices
                candidate_windows.append((current_span, left, right))
                
            # Shrink from left
            l_val, l_div = events[left]
            count[l_div] -= 1
            if count[l_div] == 0:
                unique_divs -= 1
            left += 1
            
            if time.time() > deadline:
                break
                
        if time.time() > deadline:
            break

    if not candidate_windows:
        # Fallback: just return divisors
        return [int(d) for d in divisors]

    # Sort candidate windows by span
    candidate_windows.sort(key=lambda x: x[0])
    
    # Try to find a valid distinct assignment for the best windows
    # We limit the number of windows to try to stay within time
    max_windows_to_try = 100
    
    for span, l_idx, r_idx in candidate_windows[:max_windows_to_try]:
        if time.time() > deadline:
            break
            
        L = events[l_idx][0]
        R = events[r_idx][0]
        
        # Extract candidates for each divisor in this window
        # We need to pick one value for each divisor i such that value % d_i == 0
        # and all values are distinct.
        
        # Gather all events in [L, R]
        window_events = []
        for i in range(l_idx, r_idx + 1):
            v, d_idx = events[i]
            if L <= v <= R:
                window_events.append((v, d_idx))
                
        # Group by divisor
        candidates = [[] for _ in range(n)]
        for v, d_idx in window_events:
            candidates[d_idx].append(v)
            
        # Check if any divisor has no candidates (shouldn't happen by construction of window)
        if any(len(c) == 0 for c in candidates):
            continue
            
        # Backtracking to find distinct assignment
        # Order divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
            return False
        
        if backtrack(0):
            # We found a valid assignment
            # Ensure values are positive and <= max_val
            # The values are from the event list, which are in [1, 2*period-1]
            # So they are positive.
            # Check max_val constraint
            if max(assignment) <= max_val:
                return [int(x) for x in assignment]
            # If max_val is exceeded, we might need to shift?
            # But max_val is 1,000,000 and period is 30,030.
            # Our values are < 60,000. So this check is always true.
            
    # If no window worked (unlikely), fallback
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
