"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd, isqrt

# Known minimal Egyptian fraction representations of 1 with denominators >= 5
# The goal is to minimize the number of terms.

# Let's search for representations with few terms.
# We need distinct denominators in [5, 10000000] summing reciprocals to 1.

# Let's use a systematic search approach.
# For k terms, we want 1/a1 + 1/a2 + ... + 1/ak = 1 with 5 <= a1 < a2 < ... < ak <= 10000000

# Let's think about small k:
# k=2: 1/a + 1/b = 1 => (a-1)(b-1) = 1 => a=b=2, but a,b >= 5. No solution.
# k=3: 1/a + 1/b + 1/c = 1 with a < b < c, a >= 5
#   1/a >= 1/3 => a <= 3, but a >= 5. Contradiction. No solution.
# k=4: 1/a + 1/b + 1/c + 1/d = 1 with a < b < c < d, a >= 5
#   1/a >= 1/4 => a <= 4, but a >= 5. Contradiction. No solution.
# k=5: 1/a + 1/b + 1/c + 1/d + 1/e = 1 with a < b < c < d < e, a >= 5
#   1/a >= 1/5 => a <= 5. So a = 5.
#   Then 1/b + 1/c + 1/d + 1/e = 4/5 with b < c < d < e, b >= 6
#   1/b >= 1/4 => b <= 4, but b >= 6. Contradiction. No solution.
# k=6: 1/a + ... + 1/f = 1, a >= 5
#   1/a >= 1/6 => a <= 6. So a = 5 or a = 6.
#   If a=5: 1/b + 1/c + 1/d + 1/e + 1/f = 4/5, b >= 6
#     1/b >= 1/5 => b <= 5, but b >= 6. Contradiction.
#   If a=6: 1/b + 1/c + 1/d + 1/e + 1/f = 5/6, b >= 7
#     1/b >= 1/5 => b <= 5, but b >= 7. Contradiction.
#   No solution.
# k=7: a >= 5, 1/a >= 1/7 => a <= 7. a in {5,6,7}
#   If a=5: remaining = 4/5, 5 terms, each >= 1/6. Max sum = 5/6 > 4/5. Possible.
#     But 1/b >= 1/6 => b <= 6, b >= 6, so b=6.
#     Remaining = 4/5 - 1/6 = 24/30 - 5/30 = 19/30, 4 terms, each >= 1/7.
#     Max = 4/7 ≈ 0.571, 19/30 ≈ 0.633. Contradiction. No solution.
#   If a=6: remaining = 5/6, 5 terms, each >= 1/7. Max = 5/7 ≈ 0.714, 5/6 ≈ 0.833. No.
#   If a=7: remaining = 6/7, 5 terms, each >= 1/8. Max = 5/8 = 0.625, 6/7 ≈ 0.857. No.
#   No solution.
# k=8: a >= 5, 1/a >= 1/8 => a <= 8. a in {5,6,7,8}
#   This is getting complex. Let me write a search.

def _search_k_terms(k, min_d=5, max_d=10000000, deadline=None):
    """Find k distinct denominators >= min_d whose reciprocal sum is 1."""
    if deadline and time.time() > deadline:
        return None
    
    # Lower bound: if 1/min_d * k < 1, impossible
    # Upper bound: the sum of the k smallest reciprocals starting from min_d
    # must be >= 1
    
    def _search_rec(current_sum, remaining, start_d, chosen, depth):
        if deadline and time.time() > deadline:
            return None
        if depth == 0:
            return chosen if current_sum == 1 else None
        if remaining == 0:
            return None
        
        # Pruning: if remaining < 0, or if even with the smallest possible
        # remaining denominators we can't reach the target
        # The max we can add with 'remaining' more terms starting from start_d
        # is bounded by sum of 1/start_d + 1/(start_d+1) + ... 
        # But this is hard to compute. Let's use a simpler bound.
        # Each term is at most 1/start_d, so max additional is remaining / start_d
        if remaining / start_d < 0:
            return None
        # Also, each term is at least 1/max_d, so min additional is remaining / max_d
        # But we need remaining to be achievable.
        
        # Upper bound on start_d: 1/start_d must be <= remaining (since we can't exceed)
        # Actually 1/start_d can be > remaining, that's fine, we just won't pick it.
        # But we need remaining to be representable.
        # The smallest term we can add is 1/max_d, so remaining >= remaining / max_d... 
        # That's always true. Let's think differently.
        
        # We need to pick 'remaining' distinct denominators from [start_d, max_d]
        # whose reciprocal sum is exactly 'remaining' (as a Fraction).
        
        # Bound: 1/start_d must be >= remaining / remaining = 1? No.
        # 1/start_d >= remaining / remaining... 
        # Actually: the largest single term is 1/start_d. If 1/start_d > remaining,
        # we can still proceed (we just won't pick start_d if it overshoots).
        # But if 1/start_d < remaining / remaining, i.e., 1/start_d < 1, always true.
        
        # Better bound: if remaining / remaining > 1/start_d, i.e., remaining > 1,
        # that's always true for our case.
        
        # Key bound: the sum of the 'remaining' smallest possible reciprocals
        # from [start_d, max_d] must be >= remaining.
        # Sum of 1/d for d from start_d to start_d+remaining-1 must be >= remaining.
        # This is a necessary condition.
        
        # Also: remaining <= remaining * (1/start_d), i.e., remaining <= remaining/start_d
        # i.e., 1 <= 1/start_d, i.e., start_d <= 1. Only true for start_d=1.
        # So this bound is not useful.
        
        # Let's just iterate and prune when current_sum + 1/d > remaining
        # or when we've exhausted options.
        
        for d in range(start_d, max_d + 1):
            if deadline and time.time() > deadline:
                return None
            term = Fraction(1, d)
            if current_sum + term > remaining:
                break
            new_sum = current_sum + term
            new_remaining = remaining - term
            result = _search_rec(new_sum, new_remaining, d + 1, chosen + [d], depth - 1)
            if result is not None:
                return result
        return None
    
    return _search_rec(Fraction(0), Fraction(1), min_d, [], k)


def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # For k < 8, we proved no solution exists.
    # Let's search for k=8 first, then k=9, etc.
    
    for k in range(8, 20):
        if time.time() > deadline:
            break
        result = _search_k_terms(k, min_d=5, max_d=10000000, deadline=deadline)
        if result is not None:
            # Verify
            assert len(result) == k
            assert len(set(result)) == k
            assert all(5 <= d <= 10000000 for d in result)
            s = sum(Fraction(1, d) for d in result)
            assert s == 1, f"Sum is {s}, not 1"
            return result
    
    # Fallback: use a known representation
    # Let's construct one manually.
    # 1/5 + 1/7 + 1/11 + 1/15 + 1/22 + 1/35 + 1/44 + 1/55 + 1/77 + 1/110 = 1?
    # Let me verify: common denominator is 11*7*5*2*11 = ... this is getting complex.
    # Let's use a different approach: start with a known identity and split terms.
    
    # Known: 1 = 1/2 + 1/3 + 1/6
    # Split 1/2 = 1/3 + 1/6, but 3 < 5.
    # Let's use: 1 = 1/5 + 4/5
    # 4/5 = 1/7 + 23/35
    # This is getting complicated. Let me use a computational approach.
    
    # Fallback: use the seed program's approach with a larger target
    # Or just return a valid (non-minimal) solution.
    
    # Let's try to find any solution with a simple greedy + adjustment.
    # Or use a known result.
    
    # Actually, let's try k=8 with a more targeted search.
    # For k=8, a=5 is forced (as shown above analysis was wrong, let me redo).
    # 1/a1 >= 1/8, so a1 <= 8.
    # If a1=5: remaining = 4/5, 7 terms from [6, 10^7]
    #   1/a2 >= (4/5)/7 = 4/35 ≈ 0.114, so a2 <= 8.
    #   a2 in {6,7,8}
    #   If a2=6: remaining = 4/5 - 1/6 = 19/30, 6 terms from [7, 10^7]
    #     1/a3 >= (19/30)/6 = 19/180 ≈ 0.106, so a3 <= 9.
    #     a3 in {7,8,9}
    #     If a3=7: remaining = 19/30 - 1/7 = 133/210 - 30/210 = 103/210, 5 terms from [8, 10^7]
    #       This is getting deep. Let's just run the search.
    
    # If we couldn't find a solution, use a fallback.
    # A simple valid solution: use the seed's approach.
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            result = reached[target]
            assert len(set(result)) == len(result)
            assert all(5 <= d <= 10000000 for d in result)
            s = sum(Fraction(1, d) for d in result)
            assert s == 1
            return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
