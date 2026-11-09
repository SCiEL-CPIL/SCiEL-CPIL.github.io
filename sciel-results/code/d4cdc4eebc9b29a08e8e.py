import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def find_two_terms(rem, min_d, max_d):
        """Find distinct b, c in [min_d, max_d] with 1/b + 1/c = rem.
        Returns (b, c) or None.
        Uses divisor-based approach: if rem = p/q, then (p*b - q)*(p*c - q) = q^2.
        """
        if rem <= 0:
            return None
        p = rem.numerator
        q = rem.denominator
        # (p*b - q)*(p*c - q) = q*q
        q2 = q * q
        # Find all divisors of q2
        divisors = []
        for i in range(1, isqrt(q2) + 1):
            if q2 % i == 0:
                divisors.append(i)
                if i != q2 // i:
                    divisors.append(q2 // i)
        
        # For each divisor d of q2, set p*b - q = d, so b = (d + q) / p
        # and p*c - q = q2/d, so c = (q2/d + q) / p
        candidates = []
        for d in divisors:
            d2 = q2 // d
            # b = (d + q) / p must be integer
            if (d + q) % p != 0:
                continue
            b = (d + q) // p
            # c = (d2 + q) / p must be integer
            if (d2 + q) % p != 0:
                continue
            c = (d2 + q) // p
            if b <= 0 or c <= 0:
                continue
            # Ensure b < c for ordering
            if b > c:
                b, c = c, b
            if b == c:
                continue
            if b < min_d or c > max_d:
                continue
            candidates.append((b, c))
        
        if candidates:
            # Return the one with smallest max denominator (or first valid)
            candidates.sort(key=lambda x: x[1])
            return candidates[0]
        return None

    def find_three_terms(rem, min_d, max_d):
        """Find distinct a, b, c in [min_d, max_d] with 1/a + 1/b + 1/c = rem."""
        # a must satisfy: 1/a < rem, so a > 1/rem
        # Also 3/a >= rem (since b,c > a means 1/b,1/c < 1/a), so a <= 3/rem
        a_min = min_d
        a_max = min(max_d, int(3 / rem.numerator * rem.denominator) + 1)
        for a in range(a_min, a_max + 1):
            rem2 = rem - Fraction(1, a)
            if rem2 <= 0:
                break
            # b must satisfy: 1/b < rem2, and 2/b >= rem2 (since c > b)
            # So b <= 2/rem2
            b_max = min(max_d, int(2 / rem2.numerator * rem2.denominator) + 1)
            pair = find_two_terms(rem2, a + 1, b_max)
            if pair is not None:
                return (a, pair[0], pair[1])
        return None

    def find_four_terms(rem, min_d, max_d):
        """Find distinct a, b, c, d in [min_d, max_d] with 1/a + 1/b + 1/c + 1/d = rem."""
        a_min = min_d
        a_max = min(max_d, int(4 / rem.numerator * rem.denominator) + 1)
        for a in range(a_min, a_max + 1):
            rem2 = rem - Fraction(1, a)
            if rem2 <= 0:
                break
            b_min = a + 1
            b_max = min(max_d, int(3 / rem2.numerator * rem2.denominator) + 1)
            for b in range(b_min, b_max + 1):
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                c_max = min(max_d, int(2 / rem3.numerator * rem3.denominator) + 1)
                pair = find_two_terms(rem3, b + 1, c_max)
                if pair is not None:
                    return (a, b, pair[0], pair[1])
        return None

    def find_five_terms(rem, min_d, max_d):
        """Find distinct a, b, c, d, e in [min_d, max_d] with 1/a + 1/b + 1/c + 1/d + 1/e = rem."""
        a_min = min_d
        a_max = min(max_d, int(5 / rem.numerator * rem.denominator) + 1)
        for a in range(a_min, a_max + 1):
            rem2 = rem - Fraction(1, a)
            if rem2 <= 0:
                break
            b_min = a + 1
            b_max = min(max_d, int(4 / rem2.numerator * rem2.denominator) + 1)
            for b in range(b_min, b_max + 1):
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                c_min = b + 1
                c_max = min(max_d, int(3 / rem3.numerator * rem3.denominator) + 1)
                for c in range(c_min, c_max + 1):
                    rem4 = rem3 - Fraction(1, c)
                    if rem4 <= 0:
                        break
                    d_max = min(max_d, int(2 / rem4.numerator * rem4.denominator) + 1)
                    pair = find_two_terms(rem4, c + 1, d_max)
                    if pair is not None:
                        return (a, b, c, pair[0], pair[1])
        return None

    # Search for minimal largest denominator M
    # Try 2-term, 3-term, 4-term, 5-term, 6-term expansions
    best_solution = None

    def find_six_terms(rem, min_d, max_d):
        """Find 6 distinct denominators in [min_d, max_d] summing to rem."""
        a_min = min_d
        a_max = min(max_d, int(6 / rem.numerator * rem.denominator) + 1)
        for a in range(a_min, a_max + 1):
            if time.time() > deadline:
                return None
            rem2 = rem - Fraction(1, a)
            if rem2 <= 0:
                break
            b_min = a + 1
            b_max = min(max_d, int(5 / rem2.numerator * rem2.denominator) + 1)
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return None
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                c_min = b + 1
                c_max = min(max_d, int(4 / rem3.numerator * rem3.denominator) + 1)
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return None
                    rem4 = rem3 - Fraction(1, c)
                    if rem4 <= 0:
                        break
                    d_min = c + 1
                    d_max = min(max_d, int(3 / rem4.numerator * rem4.denominator) + 1)
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            return None
                        rem5 = rem4 - Fraction(1, d)
                        if rem5 <= 0:
                            break
                        e_max = min(max_d, int(2 / rem5.numerator * rem5.denominator) + 1)
                        pair = find_two_terms(rem5, d + 1, e_max)
                        if pair is not None:
                            return (a, b, c, d, pair[0], pair[1])
        return None

    for M in range(2, 300):
        if time.time() > deadline:
            break
            
        # Check 2-term: 1/a + 1/b = 5/31, a < b <= M
        sol = find_two_terms(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
        
        # Check 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c <= M
        sol = find_three_terms(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
        
        # Check 4-term: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d <= M
        sol = find_four_terms(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
        
        # Check 5-term: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
        sol = find_five_terms(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
        
        # Check 6-term
        sol = find_six_terms(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
        
        if time.time() > deadline:
            break

    # Fallback: known solution
    return [8, 31, 248]
