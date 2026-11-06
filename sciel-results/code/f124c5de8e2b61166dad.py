import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    def time_ok():
        return time.time() - start_time < time_limit

    best_solution = None
    best_max_denom = float('inf')

    def update_solution(sol):
        nonlocal best_solution, best_max_denom
        if sol is None:
            return
        m = max(sol)
        if m < best_max_denom:
            best_max_denom = m
            best_solution = list(sol)

    # Function to find c, d such that 1/c + 1/d = p/q
    # Returns (c, d) with c < d, minimizing d.
    # c > min_c
    def find_min_max_cd(p, q, min_c):
        if p <= 0 or q <= 0:
            return None
        
        q2 = q * q
        # We want to minimize d.
        # d = (q^2/x + q) / p
        # d is minimized when x is maximized.
        # x must be a divisor of q^2.
        # Also x <= y => x <= q.
        # So we iterate x from q down to 1.
        
        # Optimization: Only check x if x divides q^2.
        # Iterating all x from q to 1 is O(q).
        # If q is large (e.g. 10^6), this is slow.
        # But we can break early if we find a solution.
        
        # Also, we need c > min_c.
        # c = (x + q) / p
        # So (x + q) / p > min_c => x + q > p * min_c => x > p * min_c - q.
        
        lower_x = max(1, p * min_c - q + 1)
        
        # We iterate x from q down to lower_x.
        # But we must check if x divides q^2.
        
        # To speed up, we can find divisors of q^2.
        # But generating divisors is also costly if q is large.
        # Given the time limit, and that we break early, simple iteration might be okay
        # if solutions are common.
        
        # However, for "no solution" cases, we scan all x.
        # Let's add a heuristic: if q is very large, limit the scan?
        # Or just rely on the fact that we only call this for promising (a,b).
        
        for x in range(q, lower_x - 1, -1):
            if q2 % x != 0:
                continue
            y = q2 // x
            c_num = x + q
            d_num = y + q
            if c_num % p != 0 or d_num % p != 0:
                continue
            c = c_num // p
            d = d_num // p
            if c <= 0 or d <= 0:
                continue
            if c == d:
                continue
            if c < min_c:
                # Should not happen due to lower_x, but check
                continue
            # Since we iterate x down from q, d is increasing?
            # Wait, d(x) decreases as x increases.
            # So x=q gives smallest d.
            # We found the first valid x (largest x), so this is the minimal d.
            return (c, d)
            
        return None

    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # a >= 7
    
    # Also check 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # This is equivalent to 1/a + 1/b + 1/c + 1/infinity.
    # We can handle this by checking if remainder after a,b is a unit fraction.
    
    max_a = 50
    for a in range(7, max_a + 1):
        if not time_ok():
            break
            
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
            
        # b must be > a
        # 1/b < rem_a => b > 1/rem_a
        b_min_float = 1.0 / float(rem_a)
        b_min = int(math.ceil(b_min_float))
        if b_min <= a:
            b_min = a + 1
            
        # Upper bound for b?
        # If b is too large, c and d will be large.
        # But we want to minimize max(d).
        # Let's scan b up to a reasonable limit.
        # If b is large, 1/b is small, rem_b is close to rem_a.
        # c, d will be large.
        # We can stop if the theoretical minimum d exceeds best_max_denom.
        # Min d for given b:
        # rem_b = p/q.
        # Max x is q.
        # c_min = (q+q)/p = 2q/p.
        # d_min = (q+q)/p = 2q/p.
        # So min d approx 2 * (denom(rem_b) / num(rem_b)).
        
        # Let's scan b from b_min to b_min + 5000 or until time runs out.
        # We can also break if b is so large that 1/b is negligible and d > best_max.
        
        b_limit = b_min + 5000
        for b in range(b_min, b_limit + 1):
            if not time_ok():
                break
                
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                # 1/b >= rem_a, so b is too small?
                # No, b_min ensures 1/b < rem_a.
                # If rem_b <= 0, it means 1/b >= rem_a.
                # This happens if b < 1/rem_a.
                # But we started at b_min = ceil(1/rem_a).
                # Due to floating point, b_min might be 1 too small.
                # If rem_b <= 0, we can't use this b.
                # As b increases, rem_b increases.
                # So we should continue, not break?
                # Wait. rem_b = rem_a - 1/b.
                # If b increases, 1/b decreases, rem_b increases.
                # So if rem_b <= 0, we need larger b.
                # But b_min is the smallest b such that 1/b < rem_a.
                # So for b >= b_min, rem_b > 0.
                # If rem_b <= 0, it's a numerical error or logic flaw.
                # Let's just continue.
                continue
                
            p = rem_b.numerator
            q = rem_b.denominator
            
            # Check if rem_b is a unit fraction (3-term solution)
            if p == 1:
                c = q
                if c > b:
                    sol = [a, b, c]
                    update_solution(sol)
                    # Don't break, maybe 4-term is better?
                    # No, 3-term has max c. 4-term has max d > c.
                    # So 3-term is always better than 4-term with same a,b.
                    # But we are iterating b.
                    # If we found a 3-term, we can skip 4-term for this b.
                    # But we still need to check other b's.
                    continue
            
            # Find c, d
            # min_c is b + 1
            res = find_min_max_cd(p, q, b + 1)
            if res:
                c, d = res
                if d < best_max_denom:
                    sol = [a, b, c, d]
                    update_solution(sol)

    # Fallback if no solution found (should not happen)
    if best_solution is None:
        best_solution = [8, 28, 1736]

    return best_solution
