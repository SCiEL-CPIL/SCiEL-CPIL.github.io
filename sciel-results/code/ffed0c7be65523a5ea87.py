import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for d in divisors:
        period = lcm(period, d)
    
    # We know from the current best score (2.0625) that the optimal span is likely small.
    # The current program finds a solution with span 33 (raw value 33? No, score is seed/value.
    # If score is 2.0625 and seed is likely 1 or related... 
    # Actually, the problem says "search score is seed/value". 
    # If the current program returns a solution with max-min = 33, and score is 2.0625, 
    # then seed/value = 2.0625 => seed = 33 * 2.0625 = 68.0625? 
    # Or maybe the raw value is the denominator. 
    # Let's look at the current program's output.
    # The current program starts with best_solution = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39].
    # max=39, min=6, range=33.
    # If the raw value is 33, and score is 2.0625, then seed = 33 * 2.0625 = 68.0625. This is not an integer.
    # Maybe the seed is fixed at 1? Then score = 1/33 = 0.0303. That doesn't match 2.0625.
    # Maybe the score is value/seed? 33/1 = 33. No.
    # Maybe the "raw value" is not the span? "The evaluator measures the raw value (lower is better)."
    # "A raw value of 0 is impossible because the integers must be distinct."
    # "search score is seed/value".
    # If the current program has score 2.0625, and it returns a span of 33.
    # Perhaps the seed is 68? 68/33 = 2.0606. Close.
    # Perhaps the seed is 68.0625?
    # Let's assume the goal is to minimize the span (max-min).
    # The current program finds a span of 33 (or better if it refines).
    # The comment in the code says "So we only care about W < 33." implying the current best found is 33.
    # If I can find a span smaller than 33, the score will improve.
    
    # Strategy:
    # 1. Precompute masks for each value in [1, period] indicating which divisors divide it.
    # 2. For each possible window width W (starting from 1 up to 32), slide a window of size W+1 across the circle of length `period`.
    #    Actually, the window is [L, L+W]. The length is W+1. The span is W.
    #    Wait, the span is max-min. If the values are in [L, L+W], the max span is W.
    #    We want to find if there is an assignment of distinct multiples to the 10 divisors within a window of width W.
    # 3. Use a sliding window with bit counts to quickly check if all divisors are covered.
    # 4. If covered, run a fast backtracking solver to find a valid assignment.
    #    To make backtracking fast, we can use bitmasks for the "used" values and candidates.
    
    # Precomputation
    # masks[v] is a bitmask of divisors that divide v.
    # We need to handle the circular nature? 
    # The problem asks for distinct positive integers. They don't have to be within one period?
    # But if we find a solution in [L, L+W], we can shift it by k*period to be anywhere.
    # So checking L in [1, period] is sufficient for existence of a pattern.
    
    # Optimization:
    # Instead of backtracking every time we find a "covered" window, we can just check if a perfect matching exists.
    # This is a bipartite matching problem: Divisors vs Values in Window.
    # Since n=10 is small, backtracking is fine, but we need it to be fast.
    # We can order divisors by the number of candidates in the window (most constrained first).
    
    max_search_val = period + 100
    masks = [0] * (max_search_val + 1)
    for v in range(1, max_search_val + 1):
        m = 0
        for i, d in enumerate(divisors):
            if v % d == 0:
                m |= (1 << i)
        masks[v] = m

    # Precompute candidate lists for each divisor for each possible window start?
    # No, that's too much memory.
    # We will compute candidates on the fly in the backtracking function, but use the precomputed masks to quickly filter.
    # Actually, for a given window [L, L+W], the candidates for divisor i are the values v in [L, L+W] such that v % divisors[i] == 0.
    # We can precompute for each divisor i, the list of multiples in [1, max_search_val].
    # Then for a window, we can binary search to find the relevant multiples.
    
    multiples = [[] for _ in range(n)]
    for i, d in enumerate(divisors):
        for k in range(1, max_search_val // d + 1):
            multiples[i].append(k * d)

    import bisect
    
    def find_assignment_fast(L, W):
        # Get candidates for each divisor
        candidates = []
        for i in range(n):
            start = bisect.bisect_left(multiples[i], L)
            end = bisect.bisect_right(multiples[i], L + W)
            # candidates[i] = multiples[i][start:end]
            # We store as list for fast access
            candidates.append(multiples[i][start:end])
            
        # Check if any is empty
        for c in candidates:
            if not c:
                return None
            
        # Sort divisors by number of candidates (ascending) to prune early
        # We need to keep track of original indices
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        used_values = set()
        assignment = [None] * n
        
        def bt(idx):
            if idx == n:
                return True
            
            orig_i = order[idx]
            for v in candidates[orig_i]:
                if v not in used_values:
                    used_values.add(v)
                    assignment[orig_i] = v
                    if bt(idx + 1):
                        return True
                    used_values.remove(v)
                    assignment[orig_i] = None
            return False
            
        if bt(0):
            return assignment
        return None

    best_solution = [int(d) for d in divisors]
    best_range = max(best_solution) - min(best_solution)
    
    start_time = time.time()
    deadline = start_time + 55

    # Search W from 1 upwards
    # We can stop when W >= best_range
    for W in range(1, best_range):
        if time.time() > deadline:
            break
            
        # Slide window
        # We need to check if all divisors have at least one multiple in [L, L+W].
        # We maintain bit_counts for the window.
        bit_counts = [0] * n
        
        # Init for L=1
        for v in range(1, W + 2):
            m = masks[v]
            for i in range(n):
                if m & (1 << i):
                    bit_counts[i] += 1
        
        found_for_w = False
        for L in range(1, period + 1):
            if time.time() > deadline:
                break
                
            if all(c > 0 for c in bit_counts):
                # Potential match
                sol = find_assignment_fast(L, W)
                if sol is not None:
                    # Validate
                    if len(set(sol)) == n and all(1 <= x <= max_val and x % divisors[i] == 0 for i, x in enumerate(sol)):
                        return [int(x) for x in sol]
            
            # Slide
            m_remove = masks[L]
            for i in range(n):
                if m_remove & (1 << i):
                    bit_counts[i] -= 1
                    
            new_v = L + W + 1
            if new_v <= max_search_val:
                m_add = masks[new_v]
                for i in range(n):
                    if m_add & (1 << i):
                        bit_counts[i] += 1
        
    return best_solution
