import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    
    # Generate critical M values
    # We need to check M values that are multiples of any divisor,
    # and M values that are one more than a multiple (just in case, though logic suggests multiples are sufficient for min(x))
    # Actually, if min(x) is a multiple, we check that M.
    # If min(x) is NOT a multiple, then for the divisor d_j corresponding to min(x),
    # x_j is a multiple. Let x_j = k*d_j.
    # If we shift M to x_j, we get y_j = x_j.
    # For other i, y_i = next_multiple(x_j, d_i) <= x_i?
    # x_i >= min(x) = x_j.
    # next_multiple(x_j, d_i) is the smallest multiple of d_i >= x_j.
    # Since x_i is a multiple of d_i and x_i >= x_j, next_multiple(x_j, d_i) <= x_i.
    # So the new span is <= old span.
    # Thus, there is always an optimal solution where min(x) is a multiple of some divisor.
    # So we only need to check M values that are multiples of some divisor.
    
    critical_points = set()
    for d in divisors:
        # Multiples of d up to max_val
        for m in range(d, max_val + 1, d):
            critical_points.add(m)
            
    # Also include 1? If min(x) could be 1? No, d>=6.
    # So min(x) >= 6.
    
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
