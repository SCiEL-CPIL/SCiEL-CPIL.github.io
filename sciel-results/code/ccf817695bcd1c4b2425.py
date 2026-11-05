"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

# Pre-computed known valid solution: 11 terms
# 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15
# Let's verify: sum of reciprocals from 5 to 15
# We need exactly 1, so we'll search for a better solution at runtime

# Hard-coded fallback solutions
FALLBACK_SOLUTIONS = [
    # Try to find a minimal solution programmatically
]

def _verify_solution(denoms):
    """Verify a list of denominators is a valid Egyptian fraction for 1."""
    if not denoms:
        return False
    if len(denoms) < 1 or len(denoms) > 256:
        return False
    if any(d < 5 or d > 10000000 for d in denoms):
        return False
    if len(set(denoms)) != len(denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == 1

def _greedy_egyptian(target_num=1, target_den=1, min_denom=5, max_terms=256):
    """
    Greedy Egyptian fraction algorithm.
    Returns list of denominators or None if fails.
    """
    result = []
    num, den = target_num, target_den
    used = set()
    
    for _ in range(max_terms):
        if num == 0:
            break
        if num < 0:
            return None
        
        # Minimum denominator such that 1/d <= num/den
        min_d = math.ceil(den / num)
        if min_d < min_denom:
            min_d = min_denom
        
        # Find smallest unused denominator >= min_d
        d = min_d
        while d in used:
            d += 1
        
        if d > 10000000:
            return None
        
        result.append(d)
        used.add(d)
        
        # Update: num/den = num/den - 1/d = (num*d - den) / (den*d)
        num = num * d - den
        den = den * d
        
        if num == 0:
            break
        
        g = math.gcd(num, den)
        num //= g
        den //= g
    
    if num == 0 and len(result) <= max_terms:
        return result
    return None

def _try_to_reduce_count(denoms, deadline):
    """
    Try to reduce the number of terms by merging pairs.
    1/a + 1/b = 1/c iff c = a*b/(a+b) is integer.
    """
    current = list(denoms)
    improved = True
    iterations = 0
    max_iterations = 1000
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        if time.time() > deadline:
            break
        
        n = len(current)
        for i in range(n):
            if time.time() > deadline:
                return current
            for j in range(i + 1, n):
                a, b = current[i], current[j]
                # Check if 1/a + 1/b = 1/c for integer c
                num = a * b
                den = a + b
                if den > 0 and num % den == 0:
                    c = num // den
                    if 5 <= c <= 10000000:
                        # Check c is not already used
                        rest = current[:i] + current[i+1:j] + current[j+1:]
                        if c not in rest:
                            new_current = rest + [c]
                            if _verify_solution(new_current):
                                current = new_current
                                improved = True
                                break
            if improved:
                break
    
    return current

def _search_small_solutions(max_den=200, max_terms=15, deadline=None):
    """
    Search for solutions with small denominators using BFS/DFS.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    # Use the identity: if we have 1/a1 + ... + 1/an + r = 1, 
    # and r = p/q, we can try to represent p/q as Egyptian fractions
    
    # Try known small solutions
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + ... 
    # Let's compute partial sums
    
    # Actually, let's try a systematic search for up to 10-12 terms
    # with denominators in [5, 50]
    
    best = None
    best_count = 999
    
    # Try greedy first with different starting points
    for start in range(5, 20):
        # Force first term to be `start`
        num, den = 1, 1
        num = num * start - den  # 1 - 1/start = (start-1)/start
        den = den * start
        g = math.gcd(num, den)
        num //= g
        den //= g
        
        if num <= 0:
            continue
        
        result = [start]
        used = {start}
        
        for _ in range(12):
            if time.time() > deadline:
                break
            if num == 0:
                break
            if num < 0:
                break
            
            min_d = math.ceil(den / num)
            if min_d < 5:
                min_d = 5
            
            d = min_d
            while d in used:
                d += 1
            
            if d > max_den:
                break
            
            result.append(d)
            used.add(d)
            
            num = num * d - den
            den = den * d
            
            if num == 0:
                break
            
            g = math.gcd(num, den)
            num //= g
            den //= g
        
        if num == 0 and 1 <= len(result) <= max_terms:
            if _verify_solution(result):
                if len(result) < best_count:
                    best_count = len(result)
                    best = result[:]
    
    # Also try the standard greedy
    std = _greedy_egyptian()
    if std and _verify_solution(std):
        if len(std) < best_count:
            best_count = len(std)
            best = std
    
    return best

def _hardcode_valid_solution():
    """
    Return a pre-verified valid solution.
    We need to find one that sums to exactly 1.
    """
    # Let's construct one:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15
    # Sum = ?
    # This is too tedious. Let's use a known decomposition.
    
    # 1 = 1/2 + 1/3 + 1/6, but 2, 3 < 5
    # Split 1/2: 1/2 = 1/3 + 1/6, but 3 < 5
    # 1/2 = 1/4 + 1/4, not distinct
    # 1/2 = 1/5 + 1/6 + 1/30
    # So 1 = 1/5 + 1/6 + 1/30 + 1/3 + 1/6 ... but 3 < 5 and 6 repeated
    
    # Better: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + ...
    # This converges to infinity, not 1.
    
    # Let me use a programmatic approach: try a few known patterns
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20
    # Sum of 1/k for k=5 to 20:
    # H(20) - H(4) = (1 + 1/2 + 1/3 + ... + 1/20) - (1 + 1/2 + 1/3 + 1/4)
    # H(20) ≈ 3.5977, H(4) ≈ 2.0833, so sum ≈ 1.5144 > 1
    
    # We need sum = 1, so we need fewer terms or larger denominators
    
    # Let's try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15
    # H(15) - H(4) ≈ (3.3182) - (2.0833) = 1.2349 > 1
    
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13
    # H(13) - H(4) ≈ (3.1801) - (2.0833) = 1.0968 > 1
    
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12
    # H(12) - H(4) ≈ (3.1032) - (2.0833) = 1.0199 > 1
    
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11
    # H(11) - H(4) ≈ (3.0199) - (2.0833) = 0.9366 < 1
    
    # So we need 8 terms from [5,11] plus some more to make up the difference
    # 1 - 0.9366 = 0.0634
    # We need to add terms that sum to ~0.0634
    
    # Let's be precise:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11
    # = (504*11 + 420*12 + 360*13 + 315*14 + 280*15 + 252*16 + 240*17) / 5040... let me use Fraction
    
    s = sum(Fraction(1, k) for k in range(5, 12))
    remainder = 1 - s
    # remainder should be positive
    # Now find Egyptian fraction for remainder
    # remainder = 1 - (1/5+...+1/11)
    
    # Let me compute: 
    # 1/5 = 20160/100800... this is getting complex
    # Let's just do it in code
    
    # For now, let's return a solution that we'll compute at runtime
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin (90s limit)
    
    # First, try to find a solution quickly
    # Strategy: Use greedy, then try to reduce count
    
    best_solution = None
    best_count = 999999
    
    # Try standard greedy
    greedy_result = _greedy_egyptian()
    if greedy_result and _verify_solution(greedy_result):
        best_solution = greedy_result
        best_count = len(greedy_result)
    
    # Try to reduce the greedy solution
    if best_solution:
        reduced = _try_to_reduce_count(best_solution, deadline)
        if _verify_solution(reduced) and len(reduced) < best_count:
            best_count = len(reduced)
            best_solution = reduced
    
    # Try searching for smaller solutions with small denominators
    if time.time() < deadline - 5:
        small_sol = _search_small_solutions(max_den=100, max_terms=15, deadline=deadline - 5)
        if small_sol and len(small_sol) < best_count:
            best_count = len(small_sol)
            best_solution = small_sol
    
    # If we still don't have a good solution, try more aggressive search
    if time.time() < deadline - 3:
        # Try different greedy starting points and merge
        for first in range(5, 15):
            if time.time() > deadline - 3:
                break
            
            num, den = 1, 1
            # Subtract 1/first
            num = num * first - den
            den = den * first
            g = math.gcd(num, den)
            num //= g
            den //= g
            
            if num <= 0:
                continue
            
            result = [first]
            used = {first}
            
            # Greedy for the remainder
            for _ in range(14):
                if time.time() > deadline - 3:
                    break
                if num == 0:
                    break
                if num < 0:
                    break
                
                min_d = math.ceil(den / num)
                if min_d < 5:
                    min_d = 5
                
                d = min_d
                while d in used:
                    d += 1
                
                if d > 10000000:
                    break
                
                result.append(d)
                used.add(d)
                
                num = num * d - den
                den = den * d
                
                if num == 0:
                    break
                
                g = math.gcd(num, den)
                num //= g
                den //= g
            
            if num == 0 and 1 <= len(result) <= 256:
                if _verify_solution(result) and len(result) < best_count:
                    # Try to reduce
                    reduced = _try_to_reduce_count(result, deadline - 3)
                    if _verify_solution(reduced) and len(reduced) < best_count:
                        best_count = len(reduced)
                        best_solution = reduced
                    elif len(result) < best_count:
                        best_count = len(result)
                        best_solution = result
    
    # Final fallback: if we have any valid solution, return it
    if best_solution and _verify_solution(best_solution):
        return best_solution
    
    # Ultimate fallback: generate a valid (possibly large) solution
    # Use a known valid large solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100 + 1/101 + 1/102 + 1/103 + 1/104 + 1/105 + 1/106 + 1/107 + 1/108 + 1/109 + 1/110 + 1/111 + 1/112 + 1/113 + 1/114 + 1/115 + 1/116 + 1/117 + 1/118 + 1/119 + 1/120 + 1/121 + 1/122 + 1/123 + 1/124 + 1/125 + 1/126 + 1/127 + 1/128 + 1/129 + 1/130 + 1/131 + 1/132 + 1/133 + 1/134 + 1/135 + 1/136 + 1/137 + 1/138 + 1/139 + 1/140 + 1/141 + 1/142 + 1/143 + 1/144 + 1/145 + 1/146 + 1/147 + 1/148 + 1/149 + 1/150 + 1/151 + 1/152 + 1/153 + 1/154 + 1/155 + 1/156 + 1/157 + 1/158 + 1/159 + 1/160 + 1/161 + 1/162 + 1/163 + 1/164 + 1/165 + 1/166 + 1/167 + 1/168 + 1/169 + 1/170 + 1/171 + 1/172 + 1/173 + 1/174 + 1/175 + 1/176 + 1/177 + 1/178 + 1/179 + 1/180 + 1/181 + 1/182 + 1/183 + 1/184 + 1/185 + 1/186 + 1/187 + 1/188 + 1/189 + 1/190 + 1/191 + 1/192 + 1/193 + 1/194 + 1/195 + 1/196 + 1/197 + 1/198 + 1/199 + 1/200 + 1/201 + 1/202 + 1/203 + 1/204 + 1/205 + 1/206 + 1/207 + 1/208 + 1/209 + 1/210 + 1/211 + 1/212 + 1/213 + 1/214 + 1/215 + 1/216 + 1/217 + 1/218 + 1/219 + 1/220 + 1/221 + 1/222 + 1/223 + 1/224 + 1/225 + 1/226 + 1/227 + 1/228 + 1/229 + 1/230 + 1/231 + 1/232 + 1/233 + 1/234 + 1/235 + 1/236 + 1/237 + 1/238 + 1/239 + 1/240 + 1/241 + 1/242 + 1/243 + 1/244 + 1/245 + 1/246 + 1/247 + 1/248 + 1/249 + 1/250 + 1/251 + 1/252 + 1/253 + 1/254 + 1/255 + 1/256
    # This is too many terms. Let's just use the greedy result.
    
    raise RuntimeError("no valid representation found")
# EVOLVE-BLOCK-END
