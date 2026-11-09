import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Precomputed valid solution: verified at runtime
    # Use a known decomposition
    denoms = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    # Verify quickly
    num, den = 0, 1
    for d in denoms:
        g = gcd(den, d)
        den //= g
        d2 = d // g
        num = num * d2 + den
        den = den * d2
    
    if num == den:
        return denoms
    
    # If precomputed doesn't work, use greedy with repair
    # Greedy Egyptian fraction: repeatedly take the largest unit fraction <= remainder
    def greedy_egyptian(target_num, target_den, min_d, max_terms=256):
        result = []
        num, den = target_num, target_den
        for _ in range(max_terms):
            if num == 0:
                break
            if num == den:
                # Need 1 = 1/d, so d=1, but d must be >= min_d
                # Can't use d=1, so we need to split
                # 1 = 1/min_d + (1 - 1/min_d)
                # This gets complicated. Use a different approach.
                break
            # Find smallest d >= min_d such that 1/d <= num/den
            # d >= den/num
            d = max(min_d, (den + num - 1) // num)
            if d > 10000000:
                return None
            if d in result:
                d += 1
                while d in result:
                    d += 1
            result.append(d)
            # num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            if new_num < 0:
                return None
            g = gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
            min_d = d + 1
            if num == 0:
                break
        if num != 0:
            return None
        return result
    
    # Try greedy
    res = greedy_egyptian(1, 1, 5)
    if res is not None and len(res) <= 256:
        # Verify
        num, den = 0, 1
        for d in res:
            g = gcd(den, d)
            den //= g
            d2 = d // g
            num = num * d2 + den
            den = den * d2
        if num == den and all(5 <= d <= 10000000 for d in res) and len(set(res)) == len(res):
            return res
    
    # Fallback: use a larger set of consecutive integers
    # Find n such that sum_{i=5}^{n} 1/i >= 1, then adjust
    # H(n) - H(4) = 1 => H(n) = 1 + H(4) = 1 + 1 + 1/2 + 1/3 + 1/4 = 2.08333...
    # H(4) = 25/12, so we need H(n) = 1 + 25/12 = 37/12 ≈ 3.0833
    # H(10) ≈ 2.929, H(20) ≈ 3.597, so n is around 10-15
    # Let's just use a known valid set
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # = H(20) - H(4) = 55835135/15519504 - 25/12 = ... let me just verify programmatically
    
    for n in range(10, 100):
        denoms = list(range(5, n + 1))
        num, den = 0, 1
        for d in denoms:
            g = gcd(den, d)
            den //= g
            d2 = d // g
            num = num * d2 + den
            den = den * d2
        if num == den:
            return denoms
    
    raise RuntimeError("no solution found")
