"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Greedy fallback solution
    def greedy_solution(target):
        denominators = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denominators.append(d)
            r -= Fraction(1, d)
        return denominators
    
    best_solution = greedy_solution(target)
    best_max = max(best_solution)
    
    # For 1/a + 1/b = p/q: (pa-q)(pb-q) = q²
    # For 1/a + 1/b + 1/c = 5/31:
    # Fix a, then 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
    # Let r = (5a-31)/(31a), then (5a-31)b - 31a)(5a-31)c - 31a) = (31a)²
    # So (5a-31)b - 31a must be a divisor of (31a)²
    
    def find_3_term_solutions(max_c):
        """Find all 3-term solutions 1/a+1/b+1/c = 5/31 with a<b<c<=max_c"""
        solutions = []
        
        # a must satisfy: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
        # Also 1/a + 1/(a+1) + 1/(a+2) > 5/31 for small a
        # And 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18 for 3 terms
        # But we can go higher with more terms; for 3 terms specifically a <= 18
        
        for a in range(7, 100):
            if time.time() > deadline:
                break
            
            # 1/b + 1/c = 5/31 - 1/a = (5a - 31) / (31a)
            num = 5 * a - 31
            den = 31 * a
            
            if num <= 0:
                continue
            
            # Reduce fraction
            g = gcd(num, den)
            num_r = num // g
            den_r = den // g
            
            # 1/b + 1/c = num_r / den_r
            # (num_r * b - den_r)(num_r * c - den_r) = den_r²
            # Let d = num_r * b - den_r, then d | den_r²
            # b = (d + den_r) / num_r must be integer
            # c = (den_r²/d + den_r) / num_r must be integer
            
            den_r_sq = den_r * den_r
            
            # Find divisors of den_r²
            # For efficiency, iterate up to sqrt
            divisors = set()
            i = 1
            while i * i <= den_r_sq:
                if den_r_sq % i == 0:
                    divisors.add(i)
                    divisors.add(den_r_sq // i)
                i += 1
            
            for d in divisors:
                # b = (d + den_r) / num_r
                if (d + den_r) % num_r != 0:
                    continue
                b = (d + den_r) // num_r
                if b <= a or b > max_c:
                    continue
                
                d2 = den_r_sq // d
                if (d2 + den_r) % num_r != 0:
                    continue
                c = (d2 + den_r) // num_r
                if c <= b or c > max_c:
                    continue
                
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    solutions.append((a, b, c))
        
        return solutions
    
    def find_4_term_solutions(max_d):
        """Find 4-term solutions 1/a+1/b+1/c+1/d = 5/31 with a<b<c<d<=max_d"""
        solutions = []
        
        # For 4 terms: a >= 7, and 4/a > 5/31 => a < 124/5 = 24.8, so a <= 24
        # b: 1/b < remaining after a, etc.
        
        for a in range(7, 25):
            if time.time() > deadline:
                break
            
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            
            # 1/b + 1/c + 1/d = r1
            # b > a, and 1/b < r1, so b > 1/r1
            # Also 3/b > r1 => b < 3/r1
            b_min = a + 1
            b_max = min(int(3 / r1) + 1, max_d)
            
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    break
                
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                
                # 1/c + 1/d = r2
                # c > b, and 1/c < r2 => c > 1/r2
                # 2/c > r2 => c < 2/r2
                c_min = b + 1
                c_max = min(int(2 / r2) + 1, max_d)
                
                # 1/c + 1/d = r2 = p/q (reduced)
                p, q = r2.numerator, r2.denominator
                q_sq = q * q
                
                # (p*c - q)(p*d - q) = q²
                # d = (q²/(pc-q) + q) / p
                # pc - q must divide q² and be positive
                # c > q/p, and pc - q < q => c < 2q/p
                c_lo = max(c_min, q // p + 1)
                c_hi = min(c_max, 2 * q // p)
                
                for c in range(c_lo, c_hi + 1):
                    denom = p * c - q
                    if denom <= 0:
                        continue
                    if (c * q) % denom != 0:
                        continue
                    d = (c * q) // denom
                    if d <= c or d > max_d:
                        continue
                    
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                        solutions.append((a, b, c, d))
        
        return solutions
    
    def find_5_term_solutions(max_e):
        """Find 5-term solutions 1/a+1/b+1/c+1/d+1/e = 5/31"""
        solutions = []
        
        for a in range(7, 20):
            if time.time() > deadline:
                break
            
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            
            b_min = a + 1
            b_max = min(int(4 / r1) + 1, max_e)
            
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    break
                
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                
                c_min = b + 1
                c_max = min(int(3 / r2) + 1, max_e)
                
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        break
                    
                    r3 = r2 - Fraction(1, c)
                    if r3 <= 0:
                        continue
                    
                    d_min = c + 1
                    d_max = min(int(2 / r3) + 1, max_e)
                    
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            break
                        
                        r4 = r3 - Fraction(1, d)
                        if r4 <= 0:
                            continue
                        
                        # 1/e = r4, so e = 1/r4 must be integer
                        if r4.denominator != 1:
                            continue
                        e = r4.numerator
                        if e <= d or e > max_e:
                            continue
                        
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == target:
                            solutions.append((a, b, c, d, e))
        
        return solutions
    
    # Search for optimal solutions
    # Try increasing bounds
    for bound in [100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
        if time.time() > deadline:
            break
        
        # 3-term solutions
        sols3 = find_3_term_solutions(bound)
        for a, b, c in sols3:
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
        
        # 4-term solutions
        sols4 = find_4_term_solutions(bound)
        for a, b, c, d in sols4:
            if d < best_max:
                best_max = d
                best_solution = [a, b, c, d]
        
        # 5-term solutions
        if best_max > bound:
            sols5 = find_5_term_solutions(bound)
            for a, b, c, d, e in sols5:
                if e < best_max:
                    best_max = e
                    best_solution = [a, b, c, d, e]
        
        # Early termination if we found a very small max
        if best_max <= 31:
            break
    
    # Final verification
    if best_solution is not None:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
        
        # Ensure constraints
        if len(best_solution) > 256 or len(set(best_solution)) != len(best_solution) or any(d < 1 or d > 10000000 for d in best_solution):
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
    
    return best_solution
# EVOLVE-BLOCK-END
