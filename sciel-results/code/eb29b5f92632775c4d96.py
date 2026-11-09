import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    prime_factors = {}
    for m in moduli:
        n = m
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        prime_factors[m] = factors

    def count_valid_residues(m, current_set):
        forbidden = {p: set() for p in prime_factors[m]}
        for (r, m2) in current_set:
            common_primes = prime_factors[m] & prime_factors[m2]
            for p in common_primes:
                forbidden[p].add(r % p)
        
        count = 1
        for p in prime_factors[m]:
            avail = p - len(forbidden[p])
            if avail <= 0:
                return 0
            count *= avail
        return count

    def find_residue(m, current_set):
        forbidden = {p: set() for p in prime_factors[m]}
        for (r, m2) in current_set:
            common_primes = prime_factors[m] & prime_factors[m2]
            for p in common_primes:
                forbidden[p].add(r % p)
        
        for r in range(m):
            ok = True
            for p in prime_factors[m]:
                if r % p in forbidden[p]:
                    ok = False
                    break
            if ok:
                return r
        return -1

    best_count = 0
    best_set = []
    
    primes = [m for m in moduli if len(prime_factors[m]) == 1 and m > 1]
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        current_set = [(random.randint(0, p-1), p) for p in primes]
        
        remaining = [m for m in moduli if m not in primes]
        
        while remaining:
            if time.time() > deadline:
                break
            
            counts = [count_valid_residues(m, current_set) for m in remaining]
            max_count = max(counts)
            
            if max_count == 0:
                break
            
            # Pick randomly among those with max count
            candidates = [m for m, c in zip(remaining, counts) if c == max_count]
            chosen_m = random.choice(candidates)
            
            r = find_residue(chosen_m, current_set)
            if r != -1:
                current_set.append((r, chosen_m))
            
            remaining.remove(chosen_m)
            
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set[:]
            
    return [[r, m] for r, m in best_set]
