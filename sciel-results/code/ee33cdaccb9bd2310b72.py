"""Combined program for Egyptian fraction of 5/31; minimizes largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_solution = None
    
    def update_best(sol):
        nonlocal best_max, best_solution
        if sol and len(sol) <= 256 and len(set(sol)) == len(sol):
            m = max(sol)
            if m < best_max:
                best_max = m
                best_solution = list(sol)
    
    # Phase 1: Exact 3-term enumeration
    # 1/a + 1/b + 1/c = 5/31, with a < b < c
    # Fix a in [7, 18] (since 1/a < 5/31 => a > 6.2, and 3/a > 5/31 => a < 9.3... 
    # actually need 1/a + 1/(a+1) + 1/(a+2) > 5/31 for feasibility)
    # More carefully: a >= 7, and we need 3/a >= 5/31 => a <= 18.6, so a in [7, 18]
    for a in range(7, 19):
        if time.time() > deadline:
            break
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        # Need 1/b + 1/c = r with b > a, c > b
        # 1/b + 1/c = r => c = b / (b*r - 1), so b*r - 1 must divide b
        # i.e., b*r - 1 | b, so b*r - 1 | b*(1) => let k = b*r - 1, then k | b
        # b = k*m for some m, and b*r - 1 = k => k*m*r - 1 = k => k*(m*r - 1) = 1
        # That's not quite right. Let me redo:
        # 1/b + 1/c = r => (b+c)/(bc) = r => b + c = r*b*c => c*(r*b - 1) = b
        # => c = b / (r*b - 1), so (r*b - 1) must divide b
        # Let r = p/q in lowest terms. Then r*b - 1 = (p*b - q)/q
        # c = b * q / (p*b - q), so (p*b - q) must divide b*q
        # Since gcd(p*b - q, q) = gcd(p*b, q) and gcd(p,q)=1, this is gcd(b, q)
        # So we need (p*b - q) | b*q
        # Let g = gcd(b, q), b = g*b', q = g*q', gcd(b', q') = 1
        # Then p*b - q = p*g*b' - g*q' = g*(p*b' - q')
        # And b*q = g*b' * g*q' = g^2 * b' * q'
        # So we need g*(p*b' - q') | g^2 * b' * q'
        # => (p*b' - q') | g * b' * q'
        # Since gcd(p*b' - q', b') = gcd(q', b') = 1 and gcd(p*b' - q', q') = gcd(p*b', q') = 1 (since gcd(p,q')=1 and gcd(b',q')=1)
        # So (p*b' - q') | g
        # Thus p*b' - q' must be a positive divisor of g = q / q'
        # Let d = p*b' - q', d | g, d > 0
        # b' = (q' + d) / p, so p | (q' + d)
        # And b = g * b' = (q/q') * (q' + d)/p = q*(q' + d) / (p * q')
        # For b to be integer: p * q' | q * (q' + d)
        # Since gcd(p, q') = 1 (as gcd(p,q)=1 and q' | q), we need p | (q' + d) and q' | q*(q'+d)
        # q' | q*(q'+d) => q' | q*d (since q' | q*(q'+d) = q*q' + q*d)
        # Since gcd(q', q) may not be 1... wait q' = q/g where g = gcd(b,q)... this is getting complex.
        
        # Simpler approach: iterate over possible b values
        # b > a, and 1/b < r => b > 1/r, so b >= ceil(1/r) + 1
        # Also 2/c < r => c > 2/r, and c > b, so we need b such that 2/b > r is not required...
        # Actually: 1/b + 1/c = r, c > b => 1/c < 1/b, so 2/b > r => b < 2/r
        # So b is in [ceil(1/r) + 1, floor(2/r) - 1] approximately, but let's just iterate
        # b > a, b >= ceil(1/r) + 1... actually 1/b < r => b > 1/r => b >= floor(1/r) + 1
        # And 1/b + 1/c = r with c > b means 1/b > r/2 => b < 2/r => b <= floor(2/r)
        
        r_num = r.numerator
        r_den = r.denominator
        
        # b must satisfy: b > a, 1/b < r (i.e., b > r_den/r_num), and 2/b > r (i.e., b < 2*r_den/r_num)
        b_min = max(a + 1, (r_den + r_num - 1) // r_num + 1)  # b > 1/r
        b_max = min((2 * r_den) // r_num, 10000000)  # b < 2/r
        if b_max < b_min:
            continue
        
        # For each b, check if c = b / (r*b - 1) is a positive integer > b
        # r*b - 1 = (r_num * b - r_den) / r_den
        # c = b * r_den / (r_num * b - r_den)
        # So (r_num * b - r_den) must divide b * r_den
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            denom = r_num * b - r_den
            if denom <= 0:
                continue
            # Check if denom divides b * r_den
            if (b * r_den) % denom != 0:
                continue
            c = (b * r_den) // denom
            if c <= b:
                continue
            if c > 10000000:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                sol = [a, b, c]
                update_best(sol)
    
    # Phase 2: Greedy to get initial upper bound (if not already found)
    def greedy_complete(r, min_den, max_den, max_terms=256):
        """Try to complete r using greedy with bounds. Return list or None."""
        if r == 0:
            return []
        values = []
        for _ in range(max_terms):
            if time.time() > deadline:
                return None
            if r == 0:
                return values
            d_min = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
            if d_min > max_den:
                return None
            d = d_min
            values.append(d)
            r -= Fraction(1, d)
            if r < 0:
                return None
            min_den = d + 1
        return values if r == 0 else None
    
    if best_max == float('inf'):
        for d1 in range(7, 30):
            if time.time() > deadline:
                break
            r1 = target - Fraction(1, d1)
            if r1 <= 0:
                continue
            sol = greedy_complete(r1, d1 + 1, 10000000)
            if sol is not None and len(sol) + 1 <= 256:
                update_best([d1] + sol)
    
    # Phase 3: Bounded DFS with strong pruning to find better solutions
    def dfs(r, min_den, max_den, current, depth, max_depth=256):
        """Bounded DFS with strong pruning."""
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        if r == 0:
            if current:
                update_best(current)
            return True
        if depth >= max_depth or r <= 0:
            return False
        
        d_start = max(min_den, (r.denominator + r.numerator - 1) // r.numerator)
        if d_start > max_den:
            return False
        
        # Pruning: remaining terms upper bound
        remaining_terms = max_depth - depth - 1
        if remaining_terms <= 0:
            return False
        if r > Fraction(remaining_terms, d_start + 1):
            return False
        
        # Lower bound: even using the largest possible denominators, we need enough sum
        # The minimum sum with remaining_terms terms starting from max_den is very small
        # Skip this for now
        
        for d in range(d_start, max_den + 1):
            if time.time() > deadline:
                return False
            new_r = r - Fraction(1, d)
            if new_r < 0:
                continue
            current.append(d)
            if dfs(new_r, d + 1, max_den, current, depth + 1, max_depth):
                current.pop()
                return True
            current.pop()
        
        return False
    
    # Try DFS with decreasing max_den bounds
    if best_max < float('inf'):
        search_start = min(best_max - 1, 300)
        for search_max in range(search_start, 50, -1):
            if time.time() > deadline:
                break
            if dfs(target, 1, search_max, [], 0, 256):
                break
    else:
        # No solution yet, try DFS with a reasonable bound
        for search_max in [500, 400, 300]:
            if time.time() > deadline:
                break
            if dfs(target, 1, search_max, [], 0, 256):
                break
    
    if best_solution is not None and len(best_solution) <= 256:
        s = sum(Fraction(1, d) for d in best_solution)
        if s == target and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback: simple greedy
    r = target
    values = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
