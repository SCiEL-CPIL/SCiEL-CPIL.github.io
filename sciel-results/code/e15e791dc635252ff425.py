import time
from fractions import Fraction

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def find_two_terms(rem, min_d, max_d):
        """Find distinct b, c in [min_d, max_d] with 1/b + 1/c = rem.
        Returns (b, c) or None."""
        if rem <= 0:
            return None
        # 1/b + 1/c = rem => c = b / (rem*b - 1)
        # So rem*b - 1 must divide b, i.e., rem*b - 1 | b
        # Let k = rem*b - 1, then k | b, so b = k*m for some m
        # rem*k*m - 1 = k => k*(rem*m - 1) = 1... wait, let me redo.
        # 1/b + 1/c = rem => c = b / (rem*b - 1)
        # c must be integer, so rem*b - 1 must divide b.
        # Let d = rem*b - 1. Then d | b.
        # b = d * q for integer q >= 1.
        # d = rem * d * q - 1 => d * (1 - rem*q) = -1 => d * (rem*q - 1) = 1
        # So rem*q - 1 = 1/d and d = 1/(rem*q - 1)
        # This means d must be a positive divisor of... hmm, this approach is tricky with fractions.
        
        # Alternative: enumerate b from min_d to max_d, compute c = b / (rem*b - 1), check if integer and in range.
        for b in range(min_d, max_d + 1):
            # c = b / (rem*b - 1)
            num = b
            den = rem * b - 1
            if den == 0:
                continue
            # Check if num/den is integer
            if den < 0:
                continue
            if num % den != 0:
                continue
            c = num // den
            if c <= b:  # Ensure distinct and b < c (we'll handle ordering)
                continue
            if c > max_d:
                continue
            if c < min_d:
                continue
            return (b, c)
        return None

    def find_three_terms(rem, min_d, max_d):
        """Find distinct a, b, c in [min_d, max_d] with 1/a + 1/b + 1/c = rem."""
        for a in range(min_d, max_d + 1):
            rem2 = rem - Fraction(1, a)
            if rem2 <= 0:
                break
            pair = find_two_terms(rem2, a + 1, max_d)
            if pair is not None:
                return (a, pair[0], pair[1])
        return None

    def find_four_terms(rem, min_d, max_d):
        """Find distinct a, b, c, d in [min_d, max_d] with 1/a + 1/b + 1/c + 1/d = rem."""
        for a in range(min_d, max_d + 1):
            rem2 = rem - Fraction(1, a)
            if rem2 <= 0:
                break
            for b in range(a + 1, max_d + 1):
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                pair = find_two_terms(rem3, b + 1, max_d)
                if pair is not None:
                    return (a, b, pair[0], pair[1])
        return None

    def find_five_terms(rem, min_d, max_d):
        """Find distinct a, b, c, d, e in [min_d, max_d] with 1/a + 1/b + 1/c + 1/d + 1/e = rem."""
        for a in range(min_d, max_d + 1):
            rem2 = rem - Fraction(1, a)
            if rem2 <= 0:
                break
            for b in range(a + 1, max_d + 1):
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                for c in range(b + 1, max_d + 1):
                    rem4 = rem3 - Fraction(1, c)
                    if rem4 <= 0:
                        break
                    pair = find_two_terms(rem4, c + 1, max_d)
                    if pair is not None:
                        return (a, b, c, pair[0], pair[1])
        return None

    # Search for minimal largest denominator M
    # Try 2-term, 3-term, 4-term, 5-term expansions
    best_solution = None

    for M in range(2, 248):
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
        
        # Check 5-term: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31, a < b < c < d < e <= M
        sol = find_five_terms(target, 1, M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return list(sol)
        
        if time.time() > deadline:
            break

    # Fallback: known solution
    return [8, 31, 248]
