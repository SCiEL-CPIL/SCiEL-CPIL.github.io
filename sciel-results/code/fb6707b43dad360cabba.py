import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill

    MIN_D = 5
    MAX_D = 10_000_000

    # Solve 1/a + 1/b = p/q for integers a, b >= min_d
    # Returns list of (a, b) pairs or empty list
    def solve_two(p, q, min_d):
        if p == 0:
            return []
        g = gcd(p, q)
        p //= g
        q //= g
        # (a - p/q)(b - p/q) = (p/q)^2
        # a + b = p(q/a + q/b)... let me use the standard approach
        # 1/a + 1/b = p/q => q(a+b) = pab => pab - qa - qb = 0
        # (pa - q)(pb - q) = q^2
        # So pa - q = d, pb - q = q^2/d for divisor d of q^2
        # a = (d + q) / p, b = (q^2/d + q) / p
        # We need p | (d+q) and p | (q^2/d + q) and a >= min_d, b >= min_d
        # d must divide q^2
        # Factorize q^2 to get all divisors
        # But q can be large... let's enumerate divisors of q^2
        # Actually, let's enumerate divisors d of q^2 where d+q is divisible by p
        # and (q*q//d + q) is divisible by p
        
        # To keep it fast, only try if q^2 isn't too huge
        if q > 100000:
            return []
        
        q2 = q * q
        results = []
        # Enumerate divisors of q^2
        i = 1
        while i * i <= q2:
            if q2 % i == 0:
                for d in [i, q2 // i]:
                    num_a = d + q
                    num_b = q2 // d + q
                    if num_a % p == 0 and num_b % p == 0:
                        a = num_a // p
                        b = num_b // p
                        if a >= min_d and b >= min_d and a <= MAX_D and b <= MAX_D:
                            if a != b:
                                results.append((a, b))
                            else:
                                results.append((a, b))
            i += 1
        return results

    # Solve 1/a + 1/b + 1/c = p/q for integers a, b, c >= min_d
    # Strategy: pick a >= min_d, then solve 1/b + 1/c = p/q - 1/a = (p - q/a)/q... 
    # Actually: 1/b + 1/c = p/q - 1/a = (pa - q)/(qa)
    # So we need pa - q > 0, i.e., a > q/p. And pa - q divides... 
    # Let's use: 1/b + 1/c = (pa - q)/(qa). Let r = pa - q, s = qa.
    # Then solve_two(r, s, min_d) gives (b, c).
    # We need a such that r > 0 and gcd(r,s) simplifies nicely.
    def solve_three(p, q, min_d):
        if p == 0:
            return []
        g = gcd(p, q)
        p //= g
        q //= g
        
        # a must satisfy: pa > q, so a > q/p
        # Also a >= min_d
        # And a <= MAX_D
        # For efficiency, bound a: since 1/b + 1/c <= 2/min_d (roughly),
        # we need p/q - 1/a <= 2/min_d, so 1/a >= p/q - 2/min_d
        # If p/q - 2/min_d <= 0, then a can be large
        # Also 1/a < p/q (since 1/b + 1/c > 0), so a > q/p
        
        lower_a = max(min_d, q // p + 1)
        # Upper bound: 1/a > 0 and we need p/q - 1/a to be representable as sum of 2 unit fractions
        # 1/a > p/q - 2/(min_d) ... not tight enough
        # Let's just iterate a from lower_a up to some reasonable bound
        # Since 1/b + 1/c >= 2/max(b,c) >= 2/MAX_D (very small)
        # And 1/b + 1/c <= 2/min_d
        # So p/q - 1/a <= 2/min_d => 1/a >= p/q - 2/min_d
        # If p/q - 2/min_d > 0: a <= q / (p/q - 2/min_d) = q*min_d / (p*min_d - 2*q)
        
        upper_a = MAX_D
        if p * min_d > 2 * q:
            upper_a = min(MAX_D, q * min_d // (p * min_d - 2 * q) + 1)
        
        if upper_a < lower_a:
            return []
        
        # Cap iterations for time safety
        max_iter = 20000
        count = 0
        results = []
        for a in range(lower_a, upper_a + 1):
            if count >= max_iter:
                break
            count += 1
            
            r = p * a - q
            s = q * a
            if r <= 0:
                continue
            
            pairs = solve_two(r, s, min_d)
            for (b, c) in pairs:
                if b >= a and c >= a:  # ensure ordering to avoid duplicates
                    results.append((a, b, c))
        
        return results

    # Solve 1/a + 1/b + 1/c + 1/d = p/q
    def solve_four(p, q, min_d):
        if p == 0:
            return []
        g = gcd(p, q)
        p //= g
        q //= g
        
        # Pick a >= min_d, then solve_three(p/q - 1/a, ...)
        # 1/b + 1/c + 1/d = p/q - 1/a = (pa - q)/(qa)
        lower_a = max(min_d, q // p + 1)
        
        # Upper bound
        # p/q - 1/a <= 3/min_d => 1/a >= p/q - 3/min_d
        upper_a = MAX_D
        if p * min_d > 3 * q:
            upper_a = min(MAX_D, q * min_d // (p * min_d - 3 * q) + 1)
        
        if upper_a < lower_a:
            return []
        
        max_iter = 10000
        count = 0
        results = []
        for a in range(lower_a, upper_a + 1):
            if count >= max_iter:
                break
            count += 1
            
            r = p * a - q
            s = q * a
            if r <= 0:
                continue
            
            triples = solve_three(r, s, min_d)
            for (b, c, d) in triples:
                if b >= a and c >= b and d >= c:
                    results.append((a, b, c, d))
        
        return results

    # Try to find representations with fewest terms
    # Start with 4 terms (since min denominator is 5, 3 terms might not work easily)
    # Actually let's try 3 first, then 4, then 5...
    
    # For 3 terms: 1/a + 1/b + 1/c = 1, a,b,c >= 5
    # a <= 3 (since 3/5 > 1)... wait, 1/5+1/5+1/5 = 3/5 < 1
    # Actually we need a >= 5, so 1/a <= 1/5, sum of 3 <= 3/5 < 1. 
    # So 3 terms with all denominators >= 5 CANNOT sum to 1!
    # We need at least 4 terms. 4/5 > 1, so 4 terms is possible.
    
    # Let's try 4 terms: 1/a + 1/b + 1/c + 1/d = 1, 5 <= a <= b <= c <= d
    # a <= 4? No, a >= 5. 4/5 > 1, so it's possible.
    # a can be 5: 1/b + 1/c + 1/d = 4/5
    # Let's search
    
    # Known identity: 1/5 + 1/6 + 1/7 + 1/8 + ... 
    # Let me just do the search
    
    # Try 4 terms
    for a in range(5, 10):  # a is bounded: 1/a + 3/max_d... a must satisfy 1/a < 1, and 1/a + 3/5 > 1 => a < 15/2 = 7.5... 
        # Actually: 1/a + 1/b + 1/c + 1/d = 1, with a <= b <= c <= d
        # 4/a >= 1 (since each term <= 1/a) => a <= 4. But a >= 5. Contradiction?
        # No: 4 terms each <= 1/a, so sum <= 4/a. For sum = 1, need 4/a >= 1, so a <= 4.
        # But a >= 5! So 4 terms with all denominators >= 5 CANNOT sum to 1 either!
        # 5/5 = 1, so 5 terms: 5/a >= 1 => a <= 5. So a = 5 is possible.
        pass
    
    # 5 terms: 1/a + 1/b + 1/c + 1/d + 1/e = 1, 5 <= a <= b <= c <= d <= e
    # 5/a >= 1 => a <= 5, so a = 5.
    # Then 1/b + 1/c + 1/d + 1/e = 4/5, with b >= 5.
    # 4/b >= 4/5 => b <= 5, so b = 5. But we need distinct denominators!
    # So b >= 6. Then 4/b >= 4/6 = 2/3 < 4/5. Possible.
    # Let's search for 5 distinct denominators >= 5 summing to 1.
    
    # 1/5 + 1/b + 1/c + 1/d + 1/e = 1
    # 1/b + 1/c + 1/d + 1/e = 4/5
    # b >= 6 (distinct from 5)
    # 4/b >= 4/5 => b <= 5. Contradiction again!
    # So 5 terms doesn't work with distinct denominators >= 5 either?
    # Wait: 4/b where b >= 6: 4/6 = 2/3 < 4/5. But we need the sum to be 4/5.
    # The max sum of 4 terms with b >= 6 is 4/6 = 2/3 < 4/5. 
    # So we can't reach 4/5 with 4 terms all >= 6.
    # Therefore 5 terms with distinct denominators >= 5 is IMPOSSIBLE.
    
    # 6 terms: 1/a+...+1/f = 1, distinct, all >= 5.
    # 6/a >= 1 => a <= 6. So a in {5, 6}.
    # Case a=5: 1/b+...+1/f = 4/5, b >= 6, distinct.
    #   5/b >= 4/5 => b <= 25/4 = 6.25, so b <= 6. b=6.
    #   1/c+1/d+1/e+1/f = 4/5 - 1/6 = 24/30 - 5/30 = 19/30, c >= 7.
    #   4/c >= 19/30 => c <= 120/19 ≈ 6.3, so c <= 6. But c >= 7. Contradiction!
    # Case a=6: 1/b+...+1/f = 5/6, b >= 7.
    #   5/b >= 5/6 => b <= 6. But b >= 7. Contradiction!
    # So 6 terms impossible too?
    
    # Let me reconsider. For n terms with distinct denominators >= 5 summing to 1:
    # The maximum sum of n terms with smallest denominators 5,6,7,...,n+4
    # We need this max sum >= 1.
    # n=4: 1/5+1/6+1/7+1/8 = 0.2+0.1667+0.1429+0.125 = 0.6345 < 1
    # n=5: +1/9 = 0.7453 < 1
    # n=6: +1/10 = 0.8453 < 1
    # n=7: +1/11 = 0.9363 < 1
    # n=8: +1/12 = 1.0196 > 1!
    # So we need at least 8 terms.
    
    # Let's search for 8 distinct denominators in [5, 10000000] summing to 1.
    # Use backtracking with pruning.
    
    def backtrack(start, remaining, count, chosen):
        if time.time() > deadline:
            return None
        if count == 8:
            if remaining == 0:
                return list(chosen)
            return None
        # Max possible sum with (8-count) terms starting from 'start'
        # 1/start + 1/(start+1) + ... + 1/(start + 8 - count - 1)
        max_sum = 0.0
        for k in range(8 - count):
            max_sum += 1.0 / (start + k)
        if max_sum < remaining - 1e-12:
            return None
        
        # Min possible: use largest denominators? No, we want to reach exactly.
        # Just try candidates
        
        # To reduce search, limit how far we go
        max_candidate = min(start + 200, 100000)  # heuristic bound
        for d in range(start, max_candidate + 1):
            if 1.0 / d > remaining + 1e-12:
                continue
            new_remaining = remaining - 1.0 / d
            if new_remaining < -1e-12:
                continue
            if new_remaining < 1e-12:
                if count + 1 == 8:
                    return list(chosen) + [d]
                # Need more terms but remaining is ~0, can't add positive terms
                continue
            # Check if remaining can be achieved with (8-count-1) terms >= d+1
            min_next = d + 1
            # Max sum of (8-count-1) terms starting at d+1
            max_next = 0.0
            for k in range(8 - count - 1):
                max_next += 1.0 / (min_next + k)
            if max_next < new_remaining - 1e-12:
                continue
            chosen.append(d)
            result = backtrack(d + 1, new_remaining, count + 1, chosen)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    # Floating point is risky for exactness. Let's use fractions.
    from fractions import Fraction
    
    def backtrack_exact(start, remaining_frac, count, chosen):
        if time.time() > deadline:
            return None
        if count == 8:
            if remaining_frac == 0:
                return list(chosen)
            return None
        
        # Pruning: max sum
        max_sum = Fraction(0)
        for k in range(8 - count):
            max_sum += Fraction(1, start + k)
        if max_sum < remaining_frac:
            return None
        
        max_candidate = min(start + 300, 200000)
        for d in range(start, max_candidate + 1):
            frac_d = Fraction(1, d)
            if frac_d > remaining_frac:
                continue
            new_remaining = remaining_frac - frac_d
            if new_remaining < 0:
                continue
            
            # Check feasibility with remaining terms
            if 8 - count - 1 > 0:
                max_next = Fraction(0)
                for k in range(8 - count - 1):
                    max_next += Fraction(1, d + 1 + k)
                if max_next < new_remaining:
                    continue
            
            chosen.append(d)
            result = backtrack_exact(d + 1, new_remaining, count + 1, chosen)
            if result is not None:
                return result
            chosen.pop()
        return None
    
    # Try to find 8-term solution
    result = backtrack_exact(5, Fraction(1, 1), 0, [])
    
    if result is not None:
        # Verify
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and len(set(result)) == 8 and all(5 <= d <= 10000000 for d in result):
            return result
    
    # If 8 fails, try 9, 10, etc.
    for n_terms in range(9, 13):
        def backtrack_n(start, remaining_frac, count, chosen, n):
            if time.time() > deadline:
                return None
            if count == n:
                if remaining_frac == 0:
                    return list(chosen)
                return None
            
            max_sum = Fraction(0)
            for k in range(n - count):
                max_sum += Fraction(1, start + k)
            if max_sum < remaining_frac:
                return None
            
            max_candidate = min(start + 300, 200000)
            for d in range(start, max_candidate + 1):
                frac_d = Fraction(1, d)
                if frac_d > remaining_frac:
                    continue
                new_remaining = remaining_frac - frac_d
                if new_remaining < 0:
                    continue
                
                if n - count - 1 > 0:
                    max_next = Fraction(0)
                    for k in range(n - count - 1):
                        max_next += Fraction(1, d + 1 + k)
                    if max_next < new_remaining:
                        continue
                
                chosen.append(d)
                result = backtrack_n(d + 1, new_remaining, count + 1, chosen, n)
                if result is not None:
                    return result
                chosen.pop()
            return None
        
        result = backtrack_n(5, Fraction(1, 1), 0, [], n_terms)
        if result is not None:
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(set(result)) == n_terms and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: use a known solution or greedy
    # Greedy algorithm (Sylvester-like but with min 5)
    chosen = []
    remaining = Fraction(1, 1)
    for _ in range(256):
        if remaining == 0:
            break
        if time.time() > deadline:
            break
        # Find largest d >= max(5, prev+1) such that 1/d <= remaining
        # and 1/d is as large as possible but <= remaining
        prev = chosen[-1] + 1 if chosen else 5
        # We want 1/d <= remaining, so d >= 1/remaining
        min_d_needed = 1  # will be constrained by prev
        # d should be >= prev and 1/d <= remaining => d >= ceil(1/remaining)
        d_min = max(prev, 1)  # placeholder
        if remaining.numerator > 0 and remaining.denominator > 0:
            d_min = max(prev, remaining.denominator // remaining.numerator + (1 if remaining.denominator % remaining.numerator else 0))
            if 1/d_min > remaining:
                d_min += 1
        
        if d_min > 10000000:
            break
        
        chosen.append(d_min)
        remaining -= Fraction(1, d_min)
    
    if remaining == 0 and 1 <= len(chosen) <= 256:
        return chosen
    
    # Last resort: return a valid (possibly non-minimal) solution
    # 1/5 + 1/5 + ... but must be distinct. 
    # Let's just return the greedy result if it's valid, else raise
    if chosen and len(chosen) <= 256 and all(5 <= d <= 10000000 for d in chosen) and len(set(chosen)) == len(chosen):
        total = sum(Fraction(1, d) for d in chosen)
        if total == 1:
            return chosen
    
    raise RuntimeError("no representation found")
