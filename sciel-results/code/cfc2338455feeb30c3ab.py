import time
import sys

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    best_span = float('inf')
    best_result = None
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Scan possible lower bounds A
    # We scan up to a reasonable limit. 
    # The span is likely small, so optimal A is likely not huge.
    # However, the pattern of multiples repeats with period LCM(divisors), which is huge.
    # But the "gap" structure is quasi-periodic or we can just scan a large range.
    # 100,000 is a safe upper bound for A to find a tight cluster.
    # If no tight cluster found, the span will be larger, but we still return the best found.
    
    max_A = 100000
    
    for A in range(1, max_A + 1):
        if time.time() > deadline:
            break
        
        # Calculate initial multiples
        x = [0] * n
        for i in range(n):
            d = divisors[i]
            # Smallest multiple of d >= A
            # ceil(A/d) * d
            # If A % d == 0, then A, else (A//d + 1) * d
            if A % d == 0:
                x[i] = A
            else:
                x[i] = (A // d + 1) * d
        
        # Check if any x[i] exceeds maximum
        # Since A is small and span is small, this is unlikely, but good to check
        # Actually, if A is close to 1000000, we might exceed.
        # But we are scanning from 1 upwards.
        
        # Resolve conflicts (distinctness)
        # We use a set to track used values
        used = set()
        valid = True
        # We process in order of increasing x to minimize pushing others up?
        # Or just standard iteration.
        # If we push a small one up, it might collide with a larger one?
        # Example: x1=2, x2=4. d1=2, d2=4.
        # If x1 collides with something, it becomes 4. Then x1=x2=4. Conflict.
        # So order matters or we need a loop.
        # Simple approach: iterate and fix.
        # Since n=10, we can just do a simple check.
        
        # Let's sort indices by x[i] to handle collisions from bottom up?
        # If we have x = [2, 4, 4], d = [2, 4, 4].
        # i=0: x=2. used={2}.
        # i=1: x=4. used={2,4}.
        # i=2: x=4. 4 in used. x = 4+4=8. used={2,4,8}.
        # Result: 2, 4, 8. Span 6.
        # If we did i=2 first: x=4. used={4}.
        # i=1: x=4. 4 in used. x=8. used={4,8}.
        # i=0: x=2. used={2,4,8}.
        # Result: 2, 4, 8. Same.
        
        # It seems the final set of values is unique for a given A regardless of resolution order,
        # because we are just finding the k-th smallest distinct multiple >= A for each divisor?
        # No, it's the smallest multiple >= A that is distinct from others.
        # The set of values is determined by the greedy "next available" logic.
        
        for i in range(n):
            v = x[i]
            # Ensure v is not in used
            while v in used:
                v += divisors[i]
            used.add(v)
            x[i] = v
            
            # Early termination if v is already too large
            # If v - A >= best_span, and min(x) >= A, then span >= v - A?
            # No, min(x) could be much larger than A.
            # But if v is very large, the max is at least v.
            # The min is at least A.
            # So span >= v - A.
            # If v - A >= best_span, this A cannot improve.
            if v - A >= best_span:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate actual span
        m = min(x)
        M = max(x)
        span = M - m
        
        if span < best_span:
            best_span = span
            best_result = x[:]
            
            # If we find a very good span, we might break early?
            # Theoretical min is 9 (10 distinct ints).
            # If best_span == 9, we can break.
            if best_span == 9:
                break
                
    # Fallback if no result found (should not happen)
    if best_result is None:
        best_result = divisors[:]
        
    return best_result
# EVOLVE-BLOCK-END
