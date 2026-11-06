from fractions import Fraction
import time
from math import gcd, isqrt

_DEADLINE = 0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 55
    
    # Try to find a small number of terms first
    # Known: 1 = 1/5 + 1/6 + 1/20 + 1/30 + 1/300 + 1/1200 + 1/4200 + 1/16800
    # Let's verify: 1/5 + 1/6 = 11/30. 1 - 11/30 = 19/30.
    # 19/30 = 1/20 + 19/60. 19/60 = 1/30 + 19/120. 19/120 = 1/300 + 19/600...
    # Actually, let's just run the DFS for small k.
    
    for k in range(8, 257):
        if time.time() > _DEADLINE:
            break
        result = _find_k_terms(k)
        if result:
            return result
            
    return greedy_solution()

def _find_k_terms(k):
    """Find k distinct denominators >= 5 summing to 1 using fast DFS."""
    global _DEADLINE
    
    def dfs(terms_rem, t_num, t_den, min_d, chosen):
        global _DEADLINE
        if time.time() > _DEADLINE:
            return None
        
        if terms_rem == 0:
            return chosen if t_num == 0 else None
        
        if terms_rem == 1:
            if t_num == 0:
                return None
            if t_den % t_num == 0:
                d = t_den // t_num
                if d >= min_d and d <= 10000000:
                    return chosen + [d]
            return None
        
        if terms_rem == 2:
            # Solve 1/a + 1/b = t_num/t_den with a > min_d, b > a
            # a > t_den/t_num
            # b = t_den*a / (t_num*a - t_den)
            # b > a => t_den*a > a*(t_num*a - t_den) => t_den > t_num*a^2 - t_den*a => t_num*a^2 - t_den*a - t_den < 0
            # Roots of t_num*a^2 - t_den*a - t_den = 0
            # a = (t_den + sqrt(t_den^2 + 4*t_num*t_den)) / (2*t_num)
            disc = t_den*t_den + 4*t_num*t_den
            sqrt_disc = isqrt(disc)
            if sqrt_disc * sqrt_disc < disc:
                sqrt_disc += 1
            
            a_max = (t_den + sqrt_disc) // (2*t_num)
            a_min = max(min_d, t_den // t_num + 1)
            
            for a in range(a_min, a_max + 1):
                if time.time() > _DEADLINE:
                    return None
                num_b = t_den * a
                den_b = t_num * a - t_den
                if den_b <= 0:
                    continue
                if num_b % den_b == 0:
                    b = num_b // den_b
                    if b > a and b <= 10000000:
                        return chosen + [a, b]
            return None

        # Pruning: max sum with smallest denominators must be >= target
        # Use float for speed
        max_s = 0.0
        for i in range(terms_rem):
            max_s += 1.0 / (min_d + i)
        if max_s < t_num / t_den - 1e-15:
            return None
        
        # d bounds
        d_lo = max(min_d, -(-t_den // t_num))  # ceil(t_den/t_num)
        d_hi = min(10000000, (terms_rem * t_den) // t_num + 2)
        
        for d in range(d_lo, d_hi + 1):
            if time.time() > _DEADLINE:
                return None
            
            new_num = t_num * d - t_den
            if new_num <= 0:
                continue
            new_den = t_den * d
            
            # Check if remaining is achievable (float for speed)
            max_rest = 0.0
            for i in range(terms_rem - 1):
                max_rest += 1.0 / (d + 1 + i)
            if new_num / new_den > max_rest + 1e-15:
                break
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs(terms_rem - 1, new_num, new_den, d + 1, chosen + [d])
            if res is not None:
                return res
        
        return None
    
    return dfs(k, 1, 1, 5, [])

def greedy_solution():
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        if len(denoms) > 256:
            break
    return denoms
