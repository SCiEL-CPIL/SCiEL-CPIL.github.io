import time
from fractions import Fraction
import random

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    primes = _sieve(1000)
    all_primes = primes
    
    best_L = 0
    best_classes = []
    
    def compute_L(covered):
        L = 0
        for i in range(1, 100001):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    def run_greedy_v2(strategy, deadline):
        nonlocal best_L, best_classes
        
        covered = bytearray(100001)
        used_primes = set()
        budget = Fraction(1)
        classes = []
        
        # Precompute 1/p as Fractions
        costs = {p: Fraction(1, p) for p in all_primes}
        
        for n in range(1, 100001):
            if time.time() > deadline:
                break
            
            if covered[n]:
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in all_primes:
                if p in used_primes:
                    continue
                cost = costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                
                # Count new coverage in a window
                window = 2000
                max_k = min(n + window, 100000)
                cnt = 0
                for k in range(n, max_k + 1, p):
                    if not covered[k]:
                        cnt += 1
                
                # Score: coverage per unit budget, with tie-breaking
                if cnt == 0:
                    continue
                    
                # Efficiency: cnt * p (since cost is 1/p)
                score = cnt * p
                
                # Add small bonus for smaller primes (they cover more frequently)
                if strategy == 0:
                    score += (1000 - p) * 0.1
                elif strategy == 1:
                    score += p * 0.1
                elif strategy == 2:
                    # Prefer primes that are close to n (residue close to 0)
                    score += (p - r) * 0.01
                else:
                    score += random.random() * 10
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget -= costs[best_p]
            classes.append([best_p, best_r])
            
            # Mark covered
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, 100001, best_p):
                covered[k] = 1
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        
        return L
    
    def run_greedy_v3(deadline):
        """Priority-based greedy: always cover the first uncovered number with the most efficient prime."""
        nonlocal best_L, best_classes
        
        covered = bytearray(100001)
        used_primes = set()
        budget = Fraction(1)
        classes = []
        
        costs = {p: Fraction(1, p) for p in all_primes}
        
        # Find first uncovered
        n = 1
        while n <= 100000 and covered[n]:
            n += 1
        
        while n <= 100000:
            if time.time() > deadline:
                break
            
            best_p = None
            best_r = 0
            best_score = -1
            
            for p in all_primes:
                if p in used_primes:
                    continue
                cost = costs[p]
                if budget < cost:
                    continue
                
                r = n % p
                
                # Look ahead to see how many new numbers this covers
                max_k = min(n + 5000, 100000)
                cnt = 0
                for k in range(n, max_k + 1, p):
                    if not covered[k]:
                        cnt += 1
                
                if cnt == 0:
                    continue
                
                # Score: maximize coverage, prefer cheaper primes
                score = cnt * p
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget -= costs[best_p]
            classes.append([best_p, best_r])
            
            # Mark covered
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, 100001, best_p):
                covered[k] = 1
            
            # Find next uncovered
            while n <= 100000 and covered[n]:
                n += 1
        
        L = compute_L(covered)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        
        return L
    
    # Run multiple strategies
    for strat in range(4):
        if time.time() > deadline - 5:
            break
        run_greedy_v2(strat, deadline)
    
    if time.time() < deadline - 10:
        run_greedy_v3(deadline)
    
    # Try random restarts with different random seeds
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            random.seed(int(time.time() * 1000) % 100000 + _)
            run_greedy_v2(3, deadline)
    
    if not best_classes:
        # Fallback: use small primes with residues 1
        best_classes = [[p, 1] for p in all_primes[:10]]
    
    return best_classes
