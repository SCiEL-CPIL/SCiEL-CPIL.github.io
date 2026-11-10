"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    best_max = None
    best_result = None
    
    def update_best(result):
        nonlocal best_max, best_result
        if not result:
            return
        max_d = max(result)
        if max_d > 10000000:
            return
        if len(result) < 1 or len(result) > 256:
            return
        if len(set(result)) != len(result):
            return
        if any(d < 1 for d in result):
            return
        if best_max is None or max_d < best_max:
            best_max = max_d
            best_result = list(result)
    
    def divisors(n):
        """Return sorted list of all positive divisors of n."""
        if n <= 0:
            return []
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        divs.sort()
        return divs
    
    def solve_two_terms(p, q, min_x, max_y=10000000):
        """Find all (x,y) with x<=y, x>=min_x, 1/x+1/y=p/q using (px-q)(py-q)=q^2."""
        results = []
        q2 = q * q
        for u in divisors(q2):
            v = q2 // u
            if (u + q) % p != 0:
                continue
            x = (u + q) // p
            if (v + q) % p != 0:
                continue
            y = (v + q) // p
            if x < 1 or y < 1:
                continue
            if x > y:
                x, y = y, x
            if x < min_x:
                continue
            if y > max_y:
                continue
            results.append((x, y))
        return results
    
    def solve_n_terms(p, q, depth, min_denom, max_denom_limit=10000000):
        """Recursively find 'depth' denominators >= min_denom summing to p/q."""
        if depth == 1:
            # Need 1/x = p/q, so x = q/p
            if q % p == 0:
                x = q // p
                if x >= min_denom and x <= max_denom_limit:
                    return [[x]]
            return []
        
        if depth == 2:
            pairs = solve_two_terms(p, q, min_denom, max_denom_limit)
            return [[x, y] for (x, y) in pairs]
        
        results = []
        # a must satisfy 1/a <= p/q => a >= ceil(q/p)
        # Also a >= min_denom
        min_a = max(min_denom, -(-q // p))
        # Upper bound: depth terms each at least 1/a, so depth/a >= p/q => a <= depth*q/p
        max_a = min(depth * q // p + 1, 500)
        
        for a in range(min_a, max_a + 1):
            if time.time() > deadline:
                break
            # rem = p/q - 1/a = (pa - q)/(qa)
            num = p * a - q
            den = q * a
            if num <= 0:
                continue
            g = gcd(num, den)
            num_r = num // g
            den_r = den // g
            # Solve for (depth-1) more terms summing to num_r/den_r, all > a
            sub = solve_n_terms(num_r, den_r, depth - 1, a + 1, max_denom_limit)
            for combo in sub:
                results.append([a] + combo)
        return results
    
    # Try 2, 3, 4, 5 terms
    for depth in range(2, 6):
        if time.time() > deadline:
            break
        solutions = solve_n_terms(5, 31, depth, 1)
        for sol in solutions:
            update_best(sol)
    
    if best_result is not None:
        return best_result
    
    # Fallback: greedy
    from fractions import Fraction
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d = -(-r.denominator // r.numerator)
        if d in values:
            d += 1
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-END
