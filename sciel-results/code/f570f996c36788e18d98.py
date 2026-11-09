# EVOLVE-BLOCK-START
import random
from math import gcd
import itertools

_LIMIT = 10**18
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def _generate_divisors(primes, exps):
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        p_pow = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_pow)
            p_pow *= p
        divs = new_divs
    return divs

def _score_from_divs(divs):
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _score(primes, exps):
    divs = _generate_divisors(primes, exps)
    return _score_from_divs(divs)

def _product(primes, exps):
    prod = 1
    for p, e in zip(primes, exps):
        prod *= p ** e
        if prod > _LIMIT:
            return prod
    return prod

def _hill_climb(primes, exps, max_steps=50):
    curr_score = _score(primes, exps)
    curr_prod = _product(primes, exps)
    
    for _ in range(max_steps):
        improved = False
        k = len(primes)
        
        # Try swapping primes
        # Only try a subset of swaps to keep it fast
        # For each i, try swapping with a few random primes not in set
        for i in range(k):
            p_old = primes[i]
            # Get a few candidates
            candidates = []
            for p in _PRIMES:
                if p != p_old and p not in primes:
                    candidates.append(p)
            
            # Shuffle and take top 5
            random.shuffle(candidates)
            for p_new in candidates[:5]:
                new_primes = primes[:]
                new_primes[i] = p_new
                new_primes.sort()
                new_exps = exps[:]
                
                new_prod = _product(new_primes, new_exps)
                if new_prod > _LIMIT:
                    continue
                
                new_score = _score(new_primes, new_exps)
                if new_score > curr_score:
                    primes = new_primes
                    exps = new_exps
                    curr_score = new_score
                    curr_prod = new_prod
                    improved = True
                    break
            if improved:
                break
        
        # Try changing exponents
        if not improved:
            for i in range(k):
                e_old = exps[i]
                e_new = 2 if e_old == 1 else 1
                new_exps = exps[:]
                new_exps[i] = e_new
                new_primes = primes[:]
                
                new_prod = _product(new_primes, new_exps)
                if new_prod > _LIMIT:
                    continue
                
                new_score = _score(new_primes, new_exps)
                if new_score > curr_score:
                    primes = new_primes
                    exps = new_exps
                    curr_score = new_score
                    curr_prod = new_prod
                    improved = True
                    break
        
        if not improved:
            break
            
    return curr_score, primes, exps

def construct():
    random.seed(12345)
    best_score = 0
    best_primes = None
    best_exps = None
    
    # Try different k
    for k in range(14, 19):
        # Initial state: first k primes, all exps 1
        primes = _PRIMES[:k]
        exps = [1] * k
        
        # Check if valid
        if _product(primes, exps) > _LIMIT:
            continue
            
        score, primes, exps = _hill_climb(primes, exps, max_steps=30)
        
        if score > best_score:
            best_score = score
            best_primes = primes[:]
            best_exps = exps[:]
            
    if best_primes is None:
        return 2
        
    n = _product(best_primes, best_exps)
    if n > _LIMIT:
        return 2
    return n

# EVOLVE-BLOCK-END
