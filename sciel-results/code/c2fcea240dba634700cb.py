import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80
    
    # Verify a known decomposition
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210
    test = [5, 6, 7, 10, 14, 35, 42, 70, 105, 210]
    s = sum(Fraction(1, d) for d in test)
    if s == 1 and all(5 <= d <= 10000000 for d in test) and len(set(test)) == len(test):
        return test
    
    # Greedy Egyptian fraction
    def greedy(target, min_d, max_terms, max_d):
        used = set()
        result = []
        rem = Fraction(target)
        for _ in range(max_terms):
            if rem == 0:
                break
            if rem <= 0:
                return None
            # d >= 1/rem
            d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            d = d_min
            while d in used:
                d += 1
            if d > max_d:
                return None
            used.add(d)
            result.append(d)
            rem -= Fraction(1, d)
            if rem < 0:
                return None
        if rem == 0:
            return result
        return None
    
    result = greedy(1, 5, 256, 10000000)
    if result is not None:
        return result
    
    raise RuntimeError("no representation found")
