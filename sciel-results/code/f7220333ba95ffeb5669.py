import time
from fractions import Fraction
import math
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    # Known solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    def check_solution(denoms):
        """Verify a solution is valid and return max denominator."""
        if len(denoms) < 1 or len(denoms) > 256:
            return None
        if len(set(denoms)) != len(denoms):
            return None
        for d in denoms:
            if d < 1 or d > 10000000:
                return None
        total = sum(Fraction(1, d) for d in denoms)
        if total != target:
            return None
        return max(denoms)
    
    # Verify initial solution
    m = check_solution(best_solution)
    if m is not None:
        best_max_denom = m
    else:
        # Fallback if initial is invalid (shouldn't happen)
        return best_solution
    
    def solve_two_terms(remainder):
        """
        Find all pairs (b, c) with b < c, b,c positive integers,
        such that 1/b + 1/c = remainder.
        Uses the identity: if remainder = p/q in lowest terms,
        then (p*b - q)(p*c - q) = q^2.
        """
        if remainder <= 0:
            return []
        
        p = remainder.numerator
        q = remainder.denominator
        
        # (p*b - q)(p*c - q) = q^2
        # Let u = p*b - q, v = p*c - q, then u*v = q^2
        # b = (u + q) / p, c = (v + q) / p
        # Need u, v positive, u < v (so b < c), u*v = q^2
        # Also b, c must be positive integers
        
        q2 = q * q
        pairs = []
        
        # Find all divisors u of q^2 where u < q^2/u (i.e., u < q)
        # Actually u < v means u < q^2/u means u^2 < q^2 means u < q
        # Wait: u*v = q^2, u < v, so u < q (since u*v=q^2 and u<v implies u<q<v)
        
        # Enumerate divisors u of q^2 with u < q
        # But also need (u+q) % p == 0 and (v+q) % p == 0 where v = q^2/u
        
        # To be efficient, iterate u from 1 to q (or sqrt(q^2) = q)
        # But q can be large. Let's think about bounds.
        
        # b = (u+q)/p > 0 always since u,q > 0
        # c = (v+q)/p > 0
        # b < c means u < v means u < q
        
        # For each divisor u of q^2 with u < q:
        #   v = q^2 // u
        #   if (u + q) % p == 0 and (v + q) % p == 0:
        #       b = (u + q) // p
        #       c = (v + q) // p
        #       if b >= 1 and c >= 1 and b < c:
        #           pairs.append((b, c))
        
        # But q can be up to ~31*something, q^2 could be large.
        # Let's just iterate u from 1 to q and check if u divides q^2.
        
        # Actually, more efficient: iterate u from 1 to min(q, int(math.isqrt(q2)))
        # For each u that divides q2, compute v = q2//u
        # Check both (u,v) and (v,u) but we only want u < v
        
        # Let me just iterate all divisors properly
        for u in range(1, q + 1):
            if q2 % u != 0:
                continue
            v = q2 // u
            if u >= v:
                continue
            if (u + q) % p != 0:
                continue
            if (v + q) % p != 0:
                continue
            b = (u + q) // p
            c = (v + q) // p
            if b >= 1 and c >= 1 and b < c:
                pairs.append((b, c))
        
        return pairs
    
    # Systematic search over first denominator d
    # 1/d < 5/31, so d > 31/5 = 6.2, so d >= 7
    # Also 1/d + 1/(d+1) + 1/(d+2) >= 5/31 for feasibility
    # And 1/d < 5/31 so d > 6.2
    
    # Upper bound on d: 1/d + 1/(d+1) + 1/(d+2) >= 5/31
    # For large d, this is approximately 3/d >= 5/31, so d <= 93
    # More precisely, let's find the max d
    
    # Also we can have more than 3 terms, but let's first try 3 terms
    # with systematic d search
    
    # For 3 terms: 1/d + 1/b + 1/c = 5/31, d < b < c
    # d >= 7 (since 1/6 = 0.1667 > 5/31 = 0.1613, wait 1/6 > 5/31? 31/5=6.2, so 1/6 > 5/31? No: 1/6 ≈ 0.1667, 5/31 ≈ 0.1613, so 1/6 > 5/31. So d must be > 6.2, i.e., d >= 7)
    
    # But wait, we could also have d where 1/d < 5/31 but we need the sum of remaining terms to work out.
    # Actually d can be as small as... well 1/d must be less than 5/31 for the remaining to be positive, OR
    # if 1/d > 5/31, that's impossible since all terms are positive.
    # So 1/d < 5/31 => d > 31/5 = 6.2 => d >= 7
    
    # Upper bound: need 1/d + 1/(d+1) + 1/(d+2) >= 5/31
    # Let's compute max d
    
    max_d = 7
    while True:
        if time.time() - start_time > time_limit - 5.0:
            break
        # Check if 1/max_d + 1/(max_d+1) + 1/(max_d+2) >= 5/31
        s = Fraction(1, max_d) + Fraction(1, max_d+1) + Fraction(1, max_d+2)
        if s < target:
            break
        max_d += 1
        if max_d > 200:
            break
    
    # Search d from 7 to max_d
    for d in range(7, max_d + 1):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        remainder = target - Fraction(1, d)
        if remainder <= 0:
            continue
        
        # Find pairs (b, c) with d < b < c, 1/b + 1/c = remainder
        pairs = solve_two_terms(remainder)
        
        for b, c in pairs:
            if b <= d:
                continue
            if c <= d:
                continue
            # Check distinctness and bounds
            if b == d or c == d:
                continue
            if b < 1 or c < 1 or b > 10000000 or c > 10000000:
                continue
            
            sol = [d, b, c]
            m = check_solution(sol)
            if m is not None and m < best_max_denom:
                best_max_denom = m
                best_solution = sol
    
    # Now try 4-term solutions with smaller max denominator
    # For 4 terms: 1/d1 + 1/d2 + 1/d3 + 1/d4 = 5/31, d1 < d2 < d3 < d4
    # This is more complex. Let's try a different approach:
    # For a given max denominator K, check if a solution exists with all terms <= K
    # But that's expensive. Instead, let's try specific small d1, d2 and solve for d3, d4
    
    # Try pairs (d1, d2) with d1 < d2, and solve 1/d3 + 1/d4 = 5/31 - 1/d1 - 1/d2
    # d1 >= 7, d2 > d1
    
    # Bound on d1: 1/d1 + 1/(d1+1) + 1/(d1+2) + 1/(d1+3) >= 5/31
    # For d1=7: 1/7+1/8+1/9+1/10 ≈ 0.1429+0.125+0.1111+0.1 = 0.479 > 5/31
    # So d1 can be larger. Let's find max d1 for 4 terms.
    
    # Actually, for 4 terms with all <= K, the max sum is 1/7+1/8+1/9+1/10 if K>=10
    # We want to minimize K, so let's try small K values.
    
    # Alternative: try all pairs (d1, d2) with d1 < d2, both small
    # d1 from 7 to some limit, d2 from d1+1 to some limit
    
    # Let's be more targeted. We know best_max_denom is currently 1736.
    # Try to find solutions with max < 1736.
    
    # For 4 terms, let's try d1 from 7 to 20, d2 from d1+1 to 30
    for d1 in range(7, 21):
        if time.time() - start_time > time_limit - 5.0:
            break
        for d2 in range(d1 + 1, 31):
            if time.time() - start_time > time_limit - 5.0:
                break
            rem2 = target - Fraction(1, d1) - Fraction(1, d2)
            if rem2 <= 0:
                continue
            # Need 1/d3 + 1/d4 = rem2 with d2 < d3 < d4
            # Use the same two-term solver
            p = rem2.numerator
            q = rem2.denominator
            q2 = q * q
            
            # Find divisors u of q^2 with u < q
            # But we also need d3 > d2
            found_pairs = []
            for u in range(1, q + 1):
                if q2 % u != 0:
                    continue
                v = q2 // u
                if u >= v:
                    continue
                if (u + q) % p != 0:
                    continue
                if (v + q) % p != 0:
                    continue
                d3 = (u + q) // p
                d4 = (v + q) // p
                if d3 > d2 and d4 > d3 and d3 >= 1 and d4 >= 1:
                    if d3 <= 10000000 and d4 <= 10000000:
                        found_pairs.append((d3, d4))
            
            for d3, d4 in found_pairs:
                sol = [d1, d2, d3, d4]
                m = check_solution(sol)
                if m is not None and m < best_max_denom:
                    best_max_denom = m
                    best_solution = sol
    
    # Try 5-term solutions with even smaller max
    # This gets expensive. Let's try a few specific combinations.
    # For 5 terms: 1/d1 + 1/d2 + 1/d3 + 1/d4 + 1/d5 = 5/31
    # Try small d1, d2, d3 and solve for d4, d5
    
    for d1 in range(7, 12):
        if time.time() - start_time > time_limit - 5.0:
            break
        for d2 in range(d1 + 1, 15):
            if time.time() - start_time > time_limit - 5.0:
                break
            for d3 in range(d2 + 1, 20):
                if time.time() - start_time > time_limit - 5.0:
                    break
                rem3 = target - Fraction(1, d1) - Fraction(1, d2) - Fraction(1, d3)
                if rem3 <= 0:
                    continue
                p = rem3.numerator
                q = rem3.denominator
                q2 = q * q
                
                for u in range(1, q + 1):
                    if q2 % u != 0:
                        continue
                    v = q2 // u
                    if u >= v:
                        continue
                    if (u + q) % p != 0:
                        continue
                    if (v + q) % p != 0:
                        continue
                    d4 = (u + q) // p
                    d5 = (v + q) // p
                    if d4 > d3 and d5 > d4 and d4 >= 1 and d5 >= 1:
                        if d4 <= 10000000 and d5 <= 10000000:
                            sol = [d1, d2, d3, d4, d5]
                            m = check_solution(sol)
                            if m is not None and m < best_max_denom:
                                best_max_denom = m
                                best_solution = sol
    
    # Try 6-term solutions
    for d1 in range(7, 10):
        if time.time() - start_time > time_limit - 5.0:
            break
        for d2 in range(d1 + 1, 12):
            if time.time() - start_time > time_limit - 5.0:
                break
            for d3 in range(d2 + 1, 15):
                if time.time() - start_time > time_limit - 5.0:
                    break
                for d4 in range(d3 + 1, 18):
                    if time.time() - start_time > time_limit - 5.0:
                        break
                    rem4 = target - Fraction(1, d1) - Fraction(1, d2) - Fraction(1, d3) - Fraction(1, d4)
                    if rem4 <= 0:
                        continue
                    p = rem4.numerator
                    q = rem4.denominator
                    q2 = q * q
                    
                    for u in range(1, q + 1):
                        if q2 % u != 0:
                            continue
                        v = q2 // u
                        if u >= v:
                            continue
                        if (u + q) % p != 0:
                            continue
                        if (v + q) % p != 0:
                            continue
                        d5 = (u + q) // p
                        d6 = (v + q) // p
                        if d5 > d4 and d6 > d5 and d5 >= 1 and d6 >= 1:
                            if d5 <= 10000000 and d6 <= 10000000:
                                sol = [d1, d2, d3, d4, d5, d6]
                                m = check_solution(sol)
                                if m is not None and m < best_max_denom:
                                    best_max_denom = m
                                    best_solution = sol
    
    # If we still have time, try a broader 3-term search with the exact method
    # and also try some 4-term with wider ranges
    if time.time() - start_time < time_limit - 10.0:
        # Extend 4-term search
        for d1 in range(7, 25):
            if time.time() - start_time > time_limit - 10.0:
                break
            for d2 in range(d1 + 1, 40):
                if time.time() - start_time > time_limit - 10.0:
                    break
                rem2 = target - Fraction(1, d1) - Fraction(1, d2)
                if rem2 <= 0:
                    continue
                p = rem2.numerator
                q = rem2.denominator
                q2 = q * q
                
                for u in range(1, q + 1):
                    if q2 % u != 0:
                        continue
                    v = q2 // u
                    if u >= v:
                        continue
                    if (u + q) % p != 0:
                        continue
                    if (v + q) % p != 0:
                        continue
                    d3 = (u + q) // p
                    d4 = (v + q) // p
                    if d3 > d2 and d4 > d3 and d3 >= 1 and d4 >= 1:
                        if d3 <= 10000000 and d4 <= 10000000:
                            sol = [d1, d2, d3, d4]
                            m = check_solution(sol)
                            if m is not None and m < best_max_denom:
                                best_max_denom = m
                                best_solution = sol
    
    return best_solution
