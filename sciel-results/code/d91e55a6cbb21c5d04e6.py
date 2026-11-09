import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    # 115s deadline to be safe (150s limit, buffer for overhead)
    deadline = start_time + 115.0
    
    # 1. Generate powerful numbers up to a small limit to find candidate N
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # For L = 10^8, max b is ~ 464. Max a is sqrt(10^8 / 1) = 10^4.
    # Total count is roughly L^(1/3) * something, which is manageable.
    
    def is_square_free(n):
        if n < 2:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    def generate_powerful(limit):
        powerful = set()
        max_b = int(limit ** (1.0/3.0)) + 1
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        return sorted(powerful)

    # Phase 1: Identify candidate N values
    # We use a moderate limit to find N with high additive energy.
    # L1 = 10^7 -> M ~ 2000. M^2 = 4M operations. Fast.
    L1 = 10**7
    p_small = generate_powerful(L1)
    
    counts_small = defaultdict(int)
    for i, a in enumerate(p_small):
        for b in p_small[i:]:
            s = a + b
            if s <= 2 * L1: # Only count sums within reasonable range
                counts_small[s] += 1
    
    # Sort candidates by count
    sorted_ns = sorted(counts_small.items(), key=lambda x: x[1], reverse=True)
    
    # Take top K candidates. These N are "hot" in the small range.
    # Hypothesis: The structure that creates many small sums also creates many large sums,
    # or at least the N itself is a good target for global search.
    candidates = [n for n, c in sorted_ns[:20]]
    
    # Ensure we include a few "generic" large candidates if needed, 
    # but the small-range hotspots are usually the best bet for high density.
    
    # Phase 2: For each candidate N, find ALL pairs (a, b) such that a+b=N
    # and a, b are powerful.
    # a and b can be up to N.
    # We need to check if a powerful number 'x' exists such that x <= N//2 
    # and N-x is also powerful.
    
    # Instead of generating all powerful numbers up to N (which might be huge 
    # if N is large, but here N <= 2*10^7 from Phase 1, so it's small),
    # we can just generate powerful numbers up to max(candidates).
    
    max_N_candidate = max(candidates) if candidates else 2
    p_full = generate_powerful(max_N_candidate)
    p_set = set(p_full)
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # Check each candidate
    for N in candidates:
        if time.time() > deadline - 5:
            break
            
        pairs = []
        # Iterate through powerful numbers a <= N/2
        for a in p_full:
            if a > N // 2:
                break
            b = N - a
            if b in p_set:
                pairs.append([a, b])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs

    # If the small-range candidates didn't yield a good result, 
    # or if we have time, we could try a different strategy.
    # However, given the time limit and the nature of the problem,
    # the "hot" N from the small range is the most likely candidate for high score.
    
    # Fallback
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
