import time
from fractions import Fraction
from math import gcd as math_gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds, safety margin for 60s limit
    
    def time_exceeded():
        return (time.time() - start_time) > deadline

    def search(N, D, min_denom, terms_left, current, used):
        """
        Recursively search for 'terms_left' distinct denominators >= min_denom
        such that their reciprocal sum is N/D.
        Returns a list of denominators or None.
        """
        if time_exceeded():
            return "TIMEOUT"
        
        if terms_left == 0:
            return current[:] if N == 0 else None
        
        if terms_left == 1:
            if N == 0:
                return None
            if D % N == 0:
                d = D // N
                if d >= min_denom and d <= 10000000 and d not in used:
                    return current + [d]
            return None
        
        # Pruning 1: Max possible sum of remaining terms
        # Sum of 1/(min_denom) + 1/(min_denom+1) + ... + 1/(min_denom+terms_left-1)
        # We need N/D <= this sum
        # Using float for quick check, then exact if close? 
        # Or just exact fraction addition for small terms_left.
        # Since terms_left is small (starting from 8), exact is fine.
        
        max_sum = Fraction(0, 1)
        for i in range(terms_left):
            d = min_denom + i
            max_sum += Fraction(1, d)
            
        if Fraction(N, D) > max_sum:
            return None
            
        # Pruning 2: Min possible sum of remaining terms
        # The denominators can be as large as 10,000,000.
        # To minimize the sum, we pick the largest possible denominators.
        # However, they must be distinct and >= min_denom.
        # The absolute minimum sum for 'terms_left' distinct denominators 
        # bounded by MAX_D is approximately terms_left / MAX_D.
        # More precisely, if we pick MAX_D, MAX_D-1, ..., MAX_D-(terms_left-1),
        # the sum is slightly larger than terms_left/MAX_D.
        # We need N/D >= min_possible_sum.
        # A safe lower bound for the sum of k distinct denominators <= MAX_D 
        # is k / MAX_D (since 1/d >= 1/MAX_D).
        # So if N/D < terms_left / 10000000, it's impossible.
        
        min_sum_bound = Fraction(terms_left, 10000000)
        if Fraction(N, D) < min_sum_bound:
            return None

        # Determine range for the next denominator d
        # 1/d <= N/D  =>  d >= D/N
        # Also d >= min_denom
        min_d = max(min_denom, (D + N - 1) // N)
        
        # Upper bound:
        # We need to be able to form the remainder N/D - 1/d with (terms_left - 1) terms.
        # The max sum of (terms_left - 1) terms starting from d+1 is roughly (terms_left-1)/(d+1).
        # So N/D - 1/d <= (terms_left - 1) / (d + 1)
        # N/D <= 1/d + (terms_left - 1)/(d+1)
        # This is a tight bound.
        # Alternatively, a simpler loose bound: 
        # N/D >= 1/d + 1/(d+1) + ... + 1/(d+terms_left-1)
        # This is hard to invert.
        # Standard bound: d <= terms_left * D / N
        # Let's use: d * N <= terms_left * D  => d <= terms_left * D / N
        max_d = min(10000000, (terms_left * D) // N + 100)
        
        if min_d > max_d:
            return None
        
        # Iterate possible denominators
        # For efficiency, we can step if the range is large, but for small terms_left,
        # the range is usually small.
        for d in range(min_d, max_d + 1):
            if time_exceeded():
                return "TIMEOUT"
            
            if d in used:
                continue
                
            # Check if 1/d <= N/D
            if N * d < D:
                continue
                
            new_N = N * d - D
            new_D = D * d
            
            # Reduce fraction
            g = math_gcd(new_N, new_D)
            if g > 1:
                new_N //= g
                new_D //= g
            
            used.add(d)
            result = search(new_N, new_D, d + 1, terms_left - 1, current + [d], used)
            used.discard(d)
            
            if result is not None and result != "TIMEOUT":
                return result
                
        return None

    best_result = None
    
    # We know k must be at least 8.
    for k in range(8, 257):
        if time_exceeded():
            break
            
        used = set()
        # Start with N=1, D=1
        result = search(1, 1, 5, k, [], used)
        
        if result is not None and result != "TIMEOUT":
            # Verify
            exact_sum = sum(Fraction(1, d) for d in result)
            if exact_sum == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result
        elif result == "TIMEOUT":
            break
            
    # Fallback: A valid solution with more terms.
    # If the search didn't find a small k solution, we can construct one.
    # But for this specific problem, k=8 or k=9 is highly likely to exist.
    # Let's provide a hardcoded valid solution just in case, or return the last found.
    # Actually, the search for k=8, 9, 10 is very fast.
    
    # If we reach here, it means no solution was found in time or valid range.
    # This should not happen for k up to 256.
    # Let's try a greedy construction as a last resort fallback.
    denoms = []
    N, D = 1, 1
    next_denom = 5
    while N != 0 and len(denoms) < 256:
        if time_exceeded():
            break
        # Find d >= next_denom such that 1/d <= N/D
        # d >= D/N
        d = max(next_denom, (D + N - 1) // N)
        if d > 10000000:
            break
        # Ensure 1/d <= N/D
        if d * N < D:
            d = D // N + 1
            if d < next_denom:
                d = next_denom
            if d > 10000000:
                break
        denoms.append(d)
        N = N * d - D
        D = D * d
        g = math_gcd(N, D)
        if g > 0:
            N //= g
            D //= g
        next_denom = d + 1
    
    if N == 0 and len(denoms) >= 1 and len(denoms) <= 256:
        if all(5 <= d <= 10000000 for d in denoms) and len(set(denoms)) == len(denoms):
            return denoms
            
    # If even greedy fails, return an empty list or raise error? 
    # The problem guarantees a solution exists.
    # Let's return a known valid set if we have one, but we don't have a hardcoded one.
    # However, the DFS should find one.
    raise RuntimeError("No valid Egyptian fraction found")
