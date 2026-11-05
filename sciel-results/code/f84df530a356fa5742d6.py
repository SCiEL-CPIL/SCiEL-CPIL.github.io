import time
import random
from math import gcd
from itertools import combinations

# Global cache for bad triples
_bad_triples_cache = None

def _get_bad_triples():
    global _bad_triples_cache
    if _bad_triples_cache is not None:
        return _bad_triples_cache
    
    bad_triples = set()
    # Precompute gcd table for speed
    # gcd_table[i][j] = gcd(i, j)
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    for a in range(1, 129):
        for b in range(a + 1, 129):
            gab = gcd_table[a][b]
            for c in range(b + 1, 129):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bad_triples.add((a, b, c))
                    
    _bad_triples_cache = bad_triples
    return bad_triples

def _is_valid_addition(S, v, bad_triples):
    # Check if adding v to S creates any bad triple
    # S is a list of selected numbers
    # We need to check all pairs in S
    # Optimization: If S is small, just iterate.
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            # Check if (a, b, v) is a bad triple
            # Ensure sorted tuple for lookup
            if a < b:
                if a < v:
                    if b < v:
                        if (a, b, v) in bad_triples:
                            return False
                    elif v < b:
                        if (a, v, b) in bad_triples:
                            return False
                elif v < a: # v < a < b
                    if (v, a, b) in bad_triples:
                        return False
            else:
                # b < a
                if b < v:
                    if a < v:
                        if (b, a, v) in bad_triples:
                            return False
                    elif v < a:
                        if (b, v, a) in bad_triples:
                            return False
                elif v < b: # v < b < a
                    if (v, b, a) in bad_triples:
                        return False
    return True

def _greedy_construct(order, bad_triples):
    S = []
    for v in order:
        if _is_valid_addition(S, v, bad_triples):
            S.append(v)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    bad_triples = _get_bad_triples()
    
    # Strategy 1: Minimum Degree Greedy
    # Calculate degree: number of bad triples each number is part of
    degrees = [0] * 129
    for (a, b, c) in bad_triples:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1
        
    # Sort numbers by degree (ascending), then by value
    order_min_degree = sorted(range(1, 129), key=lambda x: (degrees[x], x))
    best_set = _greedy_construct(order_min_degree, bad_triples)
    
    # Strategy 2: Primes First
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [i for i in range(1, 129) if is_prime(i)]
    composites = [i for i in range(1, 129) if not is_prime(i)]
    order_primes_first = primes + composites
    set2 = _greedy_construct(order_primes_first, bad_triples)
    if len(set2) > len(best_set):
        best_set = set2
        
    # Strategy 3: Randomized Restarts
    # We have time budget. Let's do some random shuffles.
    # Use a fixed seed for reproducibility in evaluation if needed, but random is fine.
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Mix of random and degree-based
        # 50% random, 50% degree-sorted with noise
        if attempts % 2 == 0:
            order = list(range(1, 129))
            rng.shuffle(order)
        else:
            # Add noise to degrees
            noisy_degrees = [(degrees[i] + rng.random(), i) for i in range(1, 129)]
            order = [x[1] for x in sorted(noisy_degrees)]
            
        current_set = _greedy_construct(order, bad_triples)
        if len(current_set) > len(best_set):
            best_set = current_set
            # If we found a very good one, we can try local search on it?
            # For now, just keep the best.
            
        # Check time
        if time.time() > deadline - 2.0:
            break
            
    return best_set
