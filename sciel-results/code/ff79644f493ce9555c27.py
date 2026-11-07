import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    
    # Generate critical M values
    # State changes when M becomes (multiple + 1)
    critical_points = {1}
    for d in divisors:
        # Multiples are d, 2d, 3d...
        # Changes happen at d+1, 2d+1, ...
        for k in range(1, max_val // d + 1):
            m = k * d
            if m < max_val:
                critical_points.add(m + 1)
            else:
                # If m == max_val, the next state would be > max_val, so we don't care about M > max_val
                # But we might care about M = max_val itself?
                # If M = max_val, x_i = max_val if max_val % d == 0, else next multiple (which is > max_val, invalid).
                # So we should include max_val if it's a valid start?
                # Actually, if M > max_val, no solution.
                # If M = max_val, x_i must be <= max_val.
                # So we only care about M such that next_multiple(M, d_i) <= max_val for all i.
                # This implies M <= max_val.
                pass
                
    critical_points = sorted(critical_points)
    
    best_span = float('inf')
    best_solution = None
    
    def next_multiple(M, d):
        return ((M + d - 1) // d) * d

    for M in critical_points:
        if time.time() > deadline:
            break
            
        if M > max_val:
            break

        candidates = []
        valid = True
        for d in divisors:
            x = next_multiple(M, d)
            if x > max_val:
                valid = False
                break
            candidates.append(x)
        
        if not valid:
            continue
            
        if len(set(candidates)) != len(candidates):
            continue
            
        span = max(candidates) - min(candidates)
        
        if span < best_span:
            best_span = span
            best_solution = candidates[:]
            
            if best_span == 9:
                break

    if best_solution is None:
        # Fallback
        best_solution = divisors[:]
        
    return [int(x) for x in best_solution]
