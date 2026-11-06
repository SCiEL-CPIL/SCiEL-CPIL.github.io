"""Optimized search for pairwise coprime integers below 512 maximizing sum 1/(512-a)."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _score(set_a):
    return sum(1.0 / (512 - a) for a in set_a)

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Precompute prime factors for all numbers 1..511
    pf = [None] * 512
    for i in range(1, 512):
        pf[i] = _prime_factors(i)
    
    # Precompute scores
    scores = [1.0 / (512 - a) for a in range(512)]
    
    # Strategy 1: All primes below 512
    primes = [p for p in range(2, 512) if _is_prime(p)]
    best_set = primes[:]
    best_score = sum(scores[p] for p in primes)
    
    # Strategy 2: Greedy from high to low (prioritize high a values)
    chosen = []
    for a in range(511, 0, -1):
        if time.time() > deadline:
            break
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    score_greedy = sum(scores[a] for a in chosen)
    if score_greedy > best_score:
        best_score = score_greedy
        best_set = chosen[:]
    
    # Strategy 3: Local search - try to swap out low-value elements for high-value composites
    # Start from the greedy solution and try to improve
    current_set = best_set[:]
    current_score = best_score
    current_pf = [pf[a] for a in current_set]
    
    # Try removing each element and adding a better one
    # Sort candidates by score descending
    candidates = list(range(511, 0, -1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_set)):
            if time.time() > deadline:
                break
            # Try removing current_set[i]
            removed = current_set[i]
            removed_pf = current_pf[i]
            new_set = current_set[:i] + current_set[i+1:]
            new_score = current_score - scores[removed]
            
            # Try adding any candidate not in new_set that is coprime to all in new_set
            for a in candidates:
                if a == removed or a in new_set:
                    continue
                # Check if a is coprime to all in new_set
                if any(gcd(a, b) != 1 for b in new_set):
                    continue
                candidate_score = new_score + scores[a]
                if candidate_score > current_score + 1e-12:
                    current_set = new_set + [a]
                    current_score = candidate_score
                    current_pf = [pf[x] for x in current_set]
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    break  # Restart the outer loop with the new set
    
    # Strategy 4: Try a different greedy approach - prioritize by 1/(512-a) but with backtracking
    # Use a more sophisticated greedy with conflict resolution
    chosen2 = []
    # Sort by score descending (which is same as a descending since 1/(512-a) increases with a)
    for a in range(511, 0, -1):
        if time.time() > deadline:
            break
        if all(gcd(a, b) == 1 for b in chosen2):
            chosen2.append(a)
    
    # Try to improve chosen2 with local search
    current_set2 = chosen2[:]
    current_score2 = sum(scores[a] for a in current_set2)
    
    improved2 = True
    while improved2 and time.time() < deadline:
        improved2 = False
        for i in range(len(current_set2)):
            if time.time() > deadline:
                break
            removed = current_set2[i]
            new_set = current_set2[:i] + current_set2[i+1:]
            new_score = current_score2 - scores[removed]
            
            for a in candidates:
                if a == removed or a in new_set:
                    continue
                if any(gcd(a, b) != 1 for b in new_set):
                    continue
                candidate_score = new_score + scores[a]
                if candidate_score > current_score2 + 1e-12:
                    current_set2 = new_set + [a]
                    current_score2 = candidate_score
                    improved2 = True
                    if current_score2 > best_score:
                        best_score = current_score2
                        best_set = current_set2[:]
                    break
    
    return best_set
# EVOLVE-BLOCK-END
