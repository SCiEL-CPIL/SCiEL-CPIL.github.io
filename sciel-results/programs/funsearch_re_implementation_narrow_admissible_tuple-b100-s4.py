import time
import random
import math

N = 5000
DEADLINE = 290.0 # Safety margin

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def _get_smallest_violated_prime(S, primes):
    n = len(S)
    # Only primes <= n matter
    relevant_primes = [p for p in primes if p <= n]
    for p in relevant_primes:
        residues = set(x % p for x in S)
        if len(residues) == p:
            return p
    return None

def _greedy_remove(S, primes, rng):
    S = list(S)
    while True:
        p = _get_smallest_violated_prime(S, primes)
        if p is None:
            break
        # Count frequencies
        freq = {}
        for x in S:
            r = x % p
            freq[r] = freq.get(r, 0) + 1
        
        # Find min freq
        min_freq = min(freq.values())
        # Get all classes with min freq
        candidates = [r for r, c in freq.items() if c == min_freq]
        # Pick one randomly
        r_to_remove = rng.choice(candidates)
        
        # Remove
        S = [x for x in S if x % p != r_to_remove]
        
    return S

def _recover(S, removed, primes):
    # S is valid. Try to add back from 'removed'
    # Maintain covered residues for each prime
    n = len(S)
    # Only primes <= n+1 potentially matter as we add elements
    # But we should check all primes <= N just in case n grows?
    # No, condition is p <= len(tuple).
    # If we add 1 element, new len is n+1. We need to check primes <= n+1.
    
    # Initialize covered sets
    covered = {}
    for p in primes:
        if p > N: break
        covered[p] = set(x % p for x in S)
        
    added_any = True
    while added_any:
        added_any = False
        # Try to add elements from 'removed'
        # We should iterate in a specific order. Smaller numbers? Random?
        # Let's sort removed to make it deterministic-ish or just iterate.
        # To be efficient, we can filter 'removed' in-place.
        
        new_removed = []
        for x in removed:
            # Check if x can be added
            new_n = len(S) + 1
            valid = True
            # Check primes <= new_n
            for p in primes:
                if p > new_n:
                    break
                r = x % p
                # If r is not in covered[p], adding x will add a new residue.
                # This is only bad if covered[p] was already full (size p-1).
                if r not in covered[p]:
                    if len(covered[p]) == p - 1:
                        valid = False
                        break
            if valid:
                # Add x
                S.append(x)
                # Update covered
                for p in primes:
                    if p > len(S):
                        break
                    covered[p].add(x % p)
                added_any = True
            else:
                new_removed.append(x)
        removed = new_removed
        
    return S

def construct():
    start_time = time.time()
    best_size = 0
    best_S = []
    
    # Multiple restarts with different random choices
    # Budget allows for several runs.
    # One run: Greedy (fast) + Recovery (slower)
    
    while time.time() - start_time < DEADLINE - 5: # Leave 5s margin
        rng = random.Random()
        S_init = list(range(N + 1))
        # Shuffle? No, greedy removal depends on counts, not order.
        # But tie-breaking uses rng.
        
        S_greedy = _greedy_remove(S_init, PRIMES, rng)
        
        # Determine removed elements
        S_set = set(S_greedy)
        removed = [x for x in range(N + 1) if x not in S_set]
        
        S_final = _recover(S_greedy, removed, PRIMES)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
            
        # If we are very close to deadline, break
        if time.time() - start_time > DEADLINE - 10:
            break
            
    return best_S
