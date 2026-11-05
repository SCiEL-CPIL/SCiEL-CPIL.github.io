"""Combined approach: meet-in-the-middle pair precomputation + systematic search."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    target_num, target_den = 5, 31
    
    # Greedy fallback solution
    def greedy_solution(target):
        denominators = []
        r = target
        while r > 0 and len(denominators) < 256:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in denominators:
                d += 1
                while d in denominators:
                    d += 1
            if d > 10000000:
                break
            denominators.append(d)
            r -= Fraction(1, d)
        return denominators if r == 0 else greedy_solution(target)
    
    best_solution = greedy_solution(target)
    best_max = max(best_solution)
    
    # Meet-in-the-middle: precompute all pairs (b,c) with b<c and their sum 1/b+1/c
    # Store in dict: sum_fraction -> list of (b, c)
    # We'll build this incrementally up to a bound
    
    def build_pair_table(max_den):
        """Build a dict mapping Fraction -> list of (b,c) for all 1<b<c<=max_den with b<c."""
        pair_map = {}
        for b in range(2, max_den + 1):
            for c in range(b + 1, max_den + 1):
                s = Fraction(1, b) + Fraction(1, c)
                if s not in pair_map:
                    pair_map[s] = []
                pair_map[s].append((b, c))
        return pair_map
    
    # For 3-term: 1/a + 1/b + 1/c = target
    # Fix a, need 1/b + 1/c = target - 1/a
    # Use pair table to look up
    
    def search_3_terms_with_table(pair_map, max_a):
        nonlocal best_solution, best_max
        for a in range(7, max_a + 1):  # a must be > 31/5 = 6.2
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            if r in pair_map:
                for b, c in pair_map[r]:
                    if b > a and c > b:
                        if c < best_max:
                            best_max = c
                            best_solution = [a, b, c]
    
    # For 4-term: 1/a + 1/b + 1/c + 1/d = target
    # Fix a, b, need 1/c + 1/d = target - 1/a - 1/b
    
    def search_4_terms_with_table(pair_map, max_a, max_b):
        nonlocal best_solution, best_max
        for a in range(7, max_a + 1):
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            for b in range(a + 1, max_b + 1):
                r = r_after_a - Fraction(1, b)
                if r <= 0:
                    continue
                if r in pair_map:
                    for c, d in pair_map[r]:
                        if c > b and d > c:
                            if d < best_max:
                                best_max = d
                                best_solution = [a, b, c, d]
    
    # For 5-term: fix a,b,c, need 1/d+1/e = target-1/a-1/b-1/c
    def search_5_terms_with_table(pair_map, max_a, max_b, max_c):
        nonlocal best_solution, best_max
        for a in range(7, max_a + 1):
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            for b in range(a + 1, max_b + 1):
                r_after_b = r_after_a - Fraction(1, b)
                if r_after_b <= 0:
                    continue
                for c in range(b + 1, max_c + 1):
                    r = r_after_b - Fraction(1, c)
                    if r <= 0:
                        continue
                    if r in pair_map:
                        for d, e in pair_map[r]:
                            if d > c and e > d:
                                if e < best_max:
                                    best_max = e
                                    best_solution = [a, b, c, d, e]
    
    # Strategy: Build pair tables at increasing sizes and search
    # For small bounds, the table is small enough to build quickly
    
    # First, try direct mathematical approach for 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    # For fixed a: 1/b + 1/c = (5a-31)/(31a)
    # (b*(5a-31) - 31a)(c*(5a-31) - 31a) = (31a)^2 ... actually let me use the standard formula
    
    def find_pairs_direct(n, d, min_val, max_val):
        """Find all pairs (x,y) with min_val < x < y <= max_val such that 1/x + 1/y = n/d"""
        pairs = []
        if n <= 0 or d <= 0:
            return pairs
        
        # (nx - d)(ny - d) = d^2
        # Let u = nx - d, v = ny - d, then u*v = d^2
        # u must be a divisor of d^2
        # x = (u + d) / n must be integer
        # y = (v + d) / n must be integer
        # u > 0 (since x > d/n), and u < d (since x < 2d/n for x < y)
        
        d_sq = d * d
        limit = isqrt(d_sq)
        
        for u in range(1, limit + 1):
            if d_sq % u != 0:
                continue
            v = d_sq // u
            
            # Check u
            if (u + d) % n != 0:
                continue
            x = (u + d) // n
            if x <= min_val or x > max_val:
                continue
            
            # Check v
            if (v + d) % n != 0:
                continue
            y = (v + d) // n
            if y <= x or y > max_val:
                continue
            
            pairs.append((x, y))
        
        return pairs
    
    # Direct search for 3-term solutions
    def search_3_terms_direct(max_a, max_den):
        nonlocal best_solution, best_max
        for a in range(7, max_a + 1):
            num = 5 * a - 31
            den = 31 * a
            if num <= 0:
                continue
            g = gcd(num, den)
            n = num // g
            d = den // g
            
            pairs = find_pairs_direct(n, d, a, max_den)
            for b, c in pairs:
                if b > a and c > b and c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
    
    # Direct search for 4-term solutions
    def search_4_terms_direct(max_a, max_b, max_den):
        nonlocal best_solution, best_max
        for a in range(7, max_a + 1):
            if time.time() > deadline:
                return
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline:
                    return
                r = r_after_a - Fraction(1, b)
                if r <= 0:
                    continue
                n, d = r.numerator, r.denominator
                pairs = find_pairs_direct(n, d, b, max_den)
                for c, d_val in pairs:
                    if c > b and d_val > c and d_val < best_max:
                        best_max = d_val
                        best_solution = [a, b, c, d_val]
    
    # Direct search for 5-term solutions
    def search_5_terms_direct(max_a, max_b, max_c, max_den):
        nonlocal best_solution, best_max
        for a in range(7, max_a + 1):
            if time.time() > deadline:
                return
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline:
                    return
                r_after_b = r_after_a - Fraction(1, b)
                if r_after_b <= 0:
                    continue
                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline:
                        return
                    r = r_after_b - Fraction(1, c)
                    if r <= 0:
                        continue
                    n, d = r.numerator, r.denominator
                    pairs = find_pairs_direct(n, d, c, max_den)
                    for d_val, e in pairs:
                        if d_val > c and e > d_val and e < best_max:
                            best_max = e
                            best_solution = [a, b, c, d_val, e]
    
    # Phase 1: Quick direct search with small bounds
    search_3_terms_direct(50, 200)
    
    if best_max > 100:
        search_4_terms_direct(30, 80, 300)
    
    if best_max > 150:
        search_5_terms_direct(20, 50, 100, 500)
    
    # Phase 2: Meet-in-the-middle with precomputed pair tables
    # Build table for denominators up to various bounds
    for table_bound in [100, 200, 400, 800, 1500]:
        if time.time() > deadline:
            break
        
        # Only build if it might help
        if best_max <= table_bound // 2:
            continue
        
        # Build pair table
        pair_map = {}
        for b in range(2, table_bound + 1):
            for c in range(b + 1, table_bound + 1):
                s = Fraction(1, b) + Fraction(1, c)
                if s not in pair_map:
                    pair_map[s] = []
                pair_map[s].append((b, c))
        
        # Search 3-term
        search_3_terms_with_table(pair_map, min(50, table_bound // 3))
        
        # Search 4-term
        if best_max > table_bound // 3:
            search_4_terms_with_table(pair_map, min(30, table_bound // 4), min(80, table_bound // 3))
        
        # Search 5-term
        if best_max > table_bound // 2:
            search_5_terms_with_table(pair_map, min(20, table_bound // 5), min(50, table_bound // 4), min(100, table_bound // 3))
        
        del pair_map  # Free memory
    
    # Phase 3: Extended direct search if still not good
    if best_max > 300 and time.time() < deadline - 5:
        search_3_terms_direct(200, best_max - 1)
    
    if best_max > 500 and time.time() < deadline - 5:
        search_4_terms_direct(50, 150, best_max - 1)
    
    # Phase 4: Try to find even better solutions with more terms
    # For 6-term: fix a,b,c,d, need 1/e+1/f
    def search_6_terms_direct(max_a, max_b, max_c, max_d, max_den):
        nonlocal best_solution, best_max
        for a in range(7, max_a + 1):
            if time.time() > deadline:
                return
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline:
                    return
                r_after_b = r_after_a - Fraction(1, b)
                if r_after_b <= 0:
                    continue
                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline:
                        return
                    r_after_c = r_after_b - Fraction(1, c)
                    if r_after_c <= 0:
                        continue
                    for d in range(c + 1, max_d + 1):
                        if time.time() > deadline:
                            return
                        r = r_after_c - Fraction(1, d)
                        if r <= 0:
                            continue
                        n, den = r.numerator, r.denominator
                        pairs = find_pairs_direct(n, den, d, max_den)
                        for e, f in pairs:
                            if e > d and f > e and f < best_max:
                                best_max = f
                                best_solution = [a, b, c, d, e, f]
    
    if best_max > 400 and time.time() < deadline - 10:
        search_6_terms_direct(15, 30, 50, 80, best_max - 1)
    
    # Final verification
    if best_solution is not None:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
        
        if len(best_solution) > 256 or len(set(best_solution)) != len(best_solution) or any(d < 1 or d > 10000000 for d in best_solution):
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
    
    return best_solution
# EVOLVE-BLOCK-END
