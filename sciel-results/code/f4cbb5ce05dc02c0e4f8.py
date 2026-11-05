"""Rewritten program: stronger search for pairwise coprime set maximizing sum 1/(512-a)."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    N = 512
    
    # Precompute primes below 512 and their prime factorizations for numbers 1..511
    # We need to check pairwise coprime, so we track prime factors of chosen numbers.
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Greedy baseline: pick in order of highest score (i.e., largest a) that is coprime to all chosen
    def greedy():
        chosen = []
        used_primes = set()
        for a in range(511, 0, -1):
            if not prime_factors[a] & used_primes:
                chosen.append(a)
                used_primes |= prime_factors[a]
        return chosen
    
    # Local search / hill climbing with random restarts
    import random
    
    best_set = greedy()
    best_score = sum(scores[a] for a in best_set)
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def valid(s):
        used = set()
        for a in s:
            if prime_factors[a] & used:
                return False
            used |= prime_factors[a]
        return True
    
    # Hill climbing: try swapping elements to improve score
    def hill_climb(initial_set, max_iters=5000):
        current = list(initial_set)
        current_score = get_score(current)
        
        # Build index
        def make_index(s):
            idx = {}
            for i, a in enumerate(s):
                idx[a] = i
            return idx
        
        # Try to improve by:
        # 1. Removing an element (if it allows adding a better one)
        # 2. Swapping an element with a better one
        # 3. Adding an element if possible
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            used_primes = set()
            for a in current:
                used_primes |= prime_factors[a]
            
            # Try adding any a not in current that is coprime
            current_set = set(current)
            for a in range(511, 0, -1):
                if a not in current_set and not (prime_factors[a] & used_primes):
                    new_set = current + [a]
                    new_score = current_score + scores[a]
                    if new_score > current_score + 1e-15:
                        current = new_set
                        current_score = new_score
                        current_set.add(a)
                        used_primes |= prime_factors[a]
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one element, add one or more
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_score = current_score - scores[removed]
                
                # Try adding best possible a to remaining
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_add = None
                best_add_score = 0
                for a in range(511, 0, -1):
                    if a not in remaining and not (prime_factors[a] & rem_used):
                        s = rem_score + scores[a]
                        if s > best_add_score:
                            best_add_score = s
                            best_add = a
                
                if best_add is not None and best_add_score > current_score + 1e-15:
                    current = remaining + [best_add]
                    current_score = best_add_score
                    improved = True
                    break
            
            if not improved:
                break
        
        return current, current_score
    
    # Multiple random restarts with perturbation
    rng = random.Random(42)
    
    # Start with greedy solution
    current_best = best_set[:]
    current_best_score = best_score
    
    # Try several random perturbations + hill climb
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Create a perturbed version: remove a few random elements, then hill climb
        perturbed = current_best[:]
        k = rng.randint(1, min(3, len(perturbed)))
        for _ in range(k):
            if len(perturbed) > 1:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        
        # Also try: start fresh with a random greedy variant
        if restart % 2 == 0:
            # Randomized greedy: shuffle candidates by score with some randomness
            candidates = list(range(511, 0, -1))
            # Add noise to scores
            noisy_scores = []
            for a in candidates:
                noise = rng.uniform(0, 0.5 * scores[a])
                noisy_scores.append((scores[a] + noise, a))
            noisy_scores.sort(reverse=True)
            
            perturbed = []
            used = set()
            for _, a in noisy_scores:
                if not (prime_factors[a] & used):
                    perturbed.append(a)
                    used |= prime_factors[a]
        
        result, score = hill_climb(perturbed, max_iters=2000)
        
        if score > current_best_score + 1e-15:
            current_best = result
            current_best_score = score
    
    # Final check
    if time.time() < deadline - 5:
        # One more pass with more iterations on the best
        result, score = hill_climb(current_best, max_iters=10000)
        if score > current_best_score + 1e-15:
            current_best = result
            current_best_score = score
    
    return current_best
# EVOLVE-BLOCK-END
