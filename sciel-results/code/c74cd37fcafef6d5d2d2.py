import math
import time
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # safety margin below 150s

def _tau_perp_squarefree(primes):
    """Compute tau_perp for squarefree n = product of primes."""
    k = len(primes)
    if k == 0:
        return 0
    if k == 1:
        return 1
    
    # Meet-in-the-middle divisor generation
    half = k // 2
    A = primes[:half]
    B = primes[half:]
    
    def gen(sub):
        res = [1]
        for p in sub:
            res = res + [x * p for x in res]
        return res
    
    da = gen(A)
    db = gen(B)
    divs = [a * b for a in da for b in db]
    divs.sort()
    
    cnt = 0
    prev = divs[0]
    for d in divs[1:]:
        if math.gcd(prev, d) == 1:
            cnt += 1
        prev = d
    return cnt

def construct():
    global _START
    deadline = _START + _DEADLINE
    
    # First 20 primes - we can use at most 15 in the product
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    max_val = 10**18
    
    # Precompute prefix products to quickly check if a subset fits
    # For a given set of primes, product must be <= 10^18
    
    best_n = 2
    best_score = 1  # n=2 has divisors [1,2], gcd(1,2)=1, so score=1
    
    def evaluate(prime_set):
        """Evaluate a set of primes, return (n, score) if valid."""
        n = 1
        for p in prime_set:
            n *= p
        if n > max_val or n < 2:
            return None, -1
        score = _tau_perp_squarefree(list(prime_set))
        return n, score
    
    # Strategy 1: Try the first 15 primes (known good baseline)
    base15 = all_primes[:15]
    n, s = evaluate(base15)
    if s > best_score:
        best_score = s
        best_n = n
    
    # Strategy 2: Try dropping 1 or 2 from first 16 and adding a later prime
    # The idea: maybe replacing a mid-range prime with a different one helps
    
    # First, let's try all subsets of size 14 and 15 from first 18 primes
    # But that's C(18,15) + C(18,14) = 816 + 3060 = 3876 evaluations
    # Each takes ~0.1-0.5s for 2^15 divisors, so ~400-2000s total
    # Too slow. Need smarter approach.
    
    # Better: Start with first 15, then try local moves (swap one prime out, one in)
    current = set(all_primes[:15])
    
    def product_of(s):
        p = 1
        for x in s:
            p *= x
        return p
    
    # Local search: try swapping one prime out and one in
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_list = sorted(current)
        cur_n = product_of(current)
        cur_score = best_score
        
        # Try removing each prime and adding a different one
        for i, p_out in enumerate(current_list):
            if time.time() > deadline - 5:
                break
            remaining = [x for x in current_list if x != p_out]
            prod_remaining = cur_n // p_out
            
            # Find primes we could add (not in current, product fits)
            for p_in in all_primes:
                if p_in in current:
                    continue
                if prod_remaining * p_in > max_val:
                    continue
                new_set = remaining + [p_in]
                new_set.sort()
                new_n = prod_remaining * p_in
                new_score = _tau_perp_squarefree(new_set)
                if new_score > best_score:
                    best_score = new_score
                    best_n = new_n
                    current = set(new_set)
                    current_list = sorted(current)
                    cur_n = new_n
                    improved = True
                    break
    
    # Also try: what if we use fewer than 15 primes but with different choices?
    # Try sizes 12, 13, 14 from first 20 primes, but only promising subsets
    # Use greedy: start with smallest primes, then try swaps
    
    for target_size in [12, 13, 14]:
        if time.time() > deadline - 20:
            break
        # Start with first target_size primes
        cand = all_primes[:target_size]
        cand_n = product_of(cand)
        if cand_n > max_val:
            continue
        cand_score = _tau_perp_squarefree(cand)
        if cand_score > best_score:
            best_score = cand_score
            best_n = cand_n
        
        # Try local improvements
        for _ in range(50):  # limited iterations
            if time.time() > deadline - 10:
                break
            improved_local = False
            cand_set = set(cand)
            for p_out in list(cand_set):
                if time.time() > deadline - 5:
                    break
                remaining = [x for x in cand if x != p_out]
                prod_rem = cand_n // p_out
                for p_in in all_primes:
                    if p_in in cand_set:
                        continue
                    if prod_rem * p_in > max_val:
                        continue
                    new_cand = remaining + [p_in]
                    new_cand.sort()
                    new_n = prod_rem * p_in
                    new_score = _tau_perp_squarefree(new_cand)
                    if new_score > cand_score:
                        cand = new_cand
                        cand_n = new_n
                        cand_score = new_score
                        cand_set = set(cand)
                        improved_local = True
                        break
                if improved_local:
                    break
            if not improved_local:
                break
        
        if cand_score > best_score:
            best_score = cand_score
            best_n = cand_n
    
    # Final: also check if any single-prime or two-prime number beats (unlikely but safe)
    if best_n < 2:
        best_n = 2
    
    return best_n
