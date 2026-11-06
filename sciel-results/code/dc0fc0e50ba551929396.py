import time
from fractions import Fraction
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    def get_greedy():
        r = Fraction(5, 31)
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    fallback = get_greedy()
    target = Fraction(5, 31)
    
    def search(rem, min_d, max_d, path):
        if time.time() > deadline:
            return None
        if rem == 0:
            return path[:]
        if min_d > max_d:
            return None
            
        lower_bound = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if lower_bound > max_d:
            return None
            
        # Prune: Max possible sum
        # Sum_{i=lower_bound}^{max_d} 1/i
        # Approx: ln(max_d) - ln(lower_bound)
        # If max_d == lower_bound, sum = 1/max_d
        if max_d == lower_bound:
            if Fraction(1, max_d) < rem:
                return None
        else:
            # Use log approximation for speed
            # rem is Fraction, convert to float for comparison?
            # rem_float = float(rem)
            # if log(max_d) - log(lower_bound) < rem_float:
            #     return None
            # This is safe if we are conservative.
            # Let's use a slightly tighter bound if range is small
            if max_d - lower_bound < 100:
                max_sum = Fraction(0, 1)
                for i in range(lower_bound, max_d + 1):
                    max_sum += Fraction(1, i)
                if max_sum < rem:
                    return None
            else:
                # Approximation
                # sum 1/i from L to U is < 1/L + ln(U/L)
                approx_max_sum = 1.0 / lower_bound + log(max_d / lower_bound)
                if approx_max_sum < float(rem):
                    return None

        for d in range(lower_bound, max_d + 1):
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
                
            path.append(d)
            res = search(new_rem, d + 1, max_d, path)
            if res is not None:
                return res
            path.pop()
            
        return None

    # Iterate M
    # We can start M from 7
    for M in range(7, 5000):
        if time.time() > deadline - 2.0:
            break
            
        res = search(target, 1, M, [])
        if res is not None:
            return res
            
    return fallback
