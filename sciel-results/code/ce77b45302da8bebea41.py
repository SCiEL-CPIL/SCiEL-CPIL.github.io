import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    deadline = time.time() + 50  # safety margin
    
    best_span = float('inf')
    best_result = None
    
    # Generate candidate L values: multiples of any divisor
    # We search L up to 200,000 to be safe, as optimal might be higher
    # but usually low.
    max_L_search = 200000
    candidate_Ls = set()
    for d in divisors:
        for m in range(d, max_L_search + 1, d):
            candidate_Ls.add(m)
    
    # Also add 1 just in case? No, L must be such that multiples exist.
    # Actually, L is just a lower bound. The min rep is >= L.
    # If L is not a multiple, the min rep is the next multiple.
    # So iterating over multiples is sufficient because if optimal min is M,
    # M is a multiple of some d, so M is in candidate_Ls.
    # If we pick L=M, we start from M.
    # If we pick L < M, we might find M too, but it's redundant.
    
    sorted_Ls = sorted(candidate_Ls)
    
    # Orderings to try
    # 0: original
    # 1: divisors ascending
    # 2: divisors descending
    # 3: based on base value (calculated per L)
    
    indices_asc = sorted(range(n), key=lambda i: divisors[i])
    indices_desc = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    
    for L in sorted_Ls:
        if time.time() > deadline:
            break
            
        # Calculate base multiples
        bases = [((L + d - 1) // d) * d for d in divisors]
        
        # If any base > max_val, skip (or check if we can bump down? No, we need >= L)
        # If base > max_val, no solution for this L with this min constraint?
        # Actually if base > max_val, we can't place it.
        if max(bases) > max_val:
            continue
            
        # Try orderings
        orderings = [
            list(range(n)),
            indices_asc,
            indices_desc,
            sorted(range(n), key=lambda i: bases[i])
        ]
        
        for order in orderings:
            used = set()
            reps = [0] * n
            valid = True
            
            for idx in order:
                d = divisors[idx]
                m = bases[idx]
                # Find smallest multiple >= L that is not in used
                # Note: bases[idx] is already >= L
                while m in used:
                    m += d
                if m > max_val:
                    valid = False
                    break
                used.add(m)
                reps[idx] = m
            
            if valid:
                # Check distinctness (implied by used set, but good to be sure)
                # used set ensures distinctness
                span = max(reps) - min(reps)
                if span < best_span:
                    best_span = span
                    best_result = reps[:]
                    if best_span == 1:
                        return [int(x) for x in best_result]
    
    if best_result is None:
        # Fallback
        best_result = divisors[:]
        
    return [int(x) for x in best_result]
