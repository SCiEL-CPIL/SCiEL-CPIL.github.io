import time
import random
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes
PRIMES = _primes(N)
NUM_PRIMES = len(PRIMES)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer
    
    best_tuple = None
    best_size = 0
    
    # We will try several random seeds and strategies
    for seed in range(50):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Initialize set S as list of all integers
        S = list(range(N + 1))
        
        # counts[p_idx][r] = number of elements in S with x % p == r
        # p_idx is index in PRIMES list
        # We only care about primes p <= len(S) eventually, but initially len(S)=5001
        # So we need counts for all primes <= 5000
        counts = []
        for p in PRIMES:
            c = [0] * p
            for x in S:
                c[x % p] += 1
            counts.append(c)
            
        # critical_primes: list of indices in PRIMES where min(counts[i]) > 0
        critical = []
        for i, p in enumerate(PRIMES):
            if min(counts[i]) > 0:
                critical.append(i)
                
        # Greedy deletion loop
        while critical:
            if time.time() > deadline:
                break
                
            # Strategy: Pick a critical prime. 
            # Smaller primes have fewer residue classes, so "emptying" one is cheaper in terms of elements removed?
            # Actually, for small p, the class size is ~N/p which is large.
            # For large p, class size is ~N/p which is small.
            # We want to remove the fewest elements.
            # So we should prioritize LARGE primes?
            # If we empty a class for large p, we remove ~N/p elements.
            # If we empty a class for small p, we remove ~N/p elements.
            # Wait, the number of elements removed is the size of the smallest class.
            # For a given set S, the smallest class for prime p has size approx |S|/p.
            # So removing elements for large p removes fewer elements.
            # Thus, we should process LARGE primes first to minimize deletions.
            
            # Let's sort critical primes by p descending (largest p first)
            # But sorting every iteration is expensive.
            # We can just pick a random one from the critical list, or the one with largest p.
            
            # Let's try picking the largest prime from critical list
            # To avoid O(C) scan for max every time, we could maintain a heap, but C is small (<669).
            # Let's just pick the one with max p.
            max_p_idx = max(critical, key=lambda i: PRIMES[i])
            
            p = PRIMES[max_p_idx]
            c = counts[max_p_idx]
            
            # Find min count residue
            min_count = min(c)
            # Get all residues with min count
            min_residues = [r for r in range(p) if c[r] == min_count]
            
            # Pick one randomly
            r_star = random.choice(min_residues)
            
            # Remove all x in S such that x % p == r_star
            to_remove = [x for x in S if x % p == r_star]
            
            if not to_remove:
                # Should not happen if min_count > 0
                critical.remove(max_p_idx)
                continue
                
            # Update S
            S = [x for x in S if x % p != r_star]
            
            # Update counts
            for x in to_remove:
                for i, pp in enumerate(PRIMES):
                    counts[i][x % pp] -= 1
                    # Check if this prime is no longer critical
                    # It is no longer critical if min(counts[i]) == 0
                    # We don't update critical list here immediately to avoid O(P^2)
                    
            # Update critical list
            # A prime is critical if min(counts[i]) > 0
            new_critical = []
            for i in critical:
                if min(counts[i]) > 0:
                    new_critical.append(i)
            critical = new_critical
            
        # Check validity and size
        if len(S) > best_size:
            # Verify admissibility
            k = len(S)
            valid = True
            for p in PRIMES:
                if p > k:
                    break
                residues = set(x % p for x in S)
                if len(residues) == p:
                    valid = False
                    break
            if valid:
                best_size = len(S)
                best_tuple = S[:]
                
    if best_tuple is None:
        # Fallback: return empty or a known small valid tuple
        # A single element is always admissible (for p=2, 1 residue covered, 1 missed)
        return [0]
        
    return best_tuple
