"""Optimized solver for the covering problem."""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _score(residues, primes, max_L=100000):
    """
    Compute the largest L such that every integer 1..L is covered at least twice.
    Uses efficient counting via difference array approach.
    """
    n = len(primes)
    # For each position x in 1..max_L, count how many residues cover it.
    # x is covered by residue r_i mod p_i if x ≡ r_i (mod p_i).
    # Instead of checking all x for all i, we can use a different approach:
    # For each i, the positions covered are r_i, r_i+p_i, r_i+2*p_i, ...
    # We need to find the first x where count(x) < 2.
    
    # Efficient approach: use numpy to create a count array
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(n):
        r = residues[i]
        p = primes[i]
        if r == 0:
            r = p  # Since we're looking at 1..max_L, 0 mod p means p, 2p, etc.
        # Positions: r, r+p, r+2p, ... <= max_L
        if r <= max_L:
            positions = np.arange(r, max_L + 1, p)
            counts[positions] += 1
    
    # Find the largest L such that all counts[1..L] >= 2
    # We can use a cumulative minimum approach
    # Find the first index where count < 2
    deficient = np.where(counts[1:] < 2)[0]
    if len(deficient) == 0:
        return max_L
    else:
        return deficient[0]  # This is the first deficient position, so L = deficient[0] - 1 + 1 = deficient[0]... wait
    
    # Actually, if deficient[0] = k, then position k+1 is the first deficient one (since counts[1:] starts at index 1)
    # So the largest L such that 1..L all have count >= 2 is k (since positions 1..k are all >= 2, and k+1 is < 2)
    # Wait: counts[1:] has indices 0..max_L-1 corresponding to positions 1..max_L
    # So deficient[0] = j means position j+1 is the first deficient one
    # So L = j (positions 1..j are all covered at least twice)
    return deficient[0]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 145 seconds, safety margin
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Initial solution: the seed program's approach
    # r_i = ((i+2)//2) % p_i
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = _score(best_residues, primes, 10000)
    
    # Try to improve with local search
    # Strategy: for each position i, try all possible residues and keep the best
    # This is expensive, so we'll do multiple passes with limited budget
    
    import random
    rng = np.random.RandomState(42)
    
    # Greedy improvement: iterate through positions, try to improve each one
    # For speed, we'll only check a subset of possible residues for each position
    # Focus on positions that are "critical" - where the coverage is thin
    
    # First, let's try a systematic approach: for each i, try a few random residues
    # and keep the best overall
    
    # More efficient: hill climbing with random restarts
    # But given time constraints, let's do a few passes of local search
    
    max_L_check = 5000  # Check up to 5000 for speed during search
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Randomly select a subset of positions to modify
        # Focus on small primes where changes have more impact
        indices = list(range(n))
        rng.shuffle(indices)
        
        improved = False
        for idx in indices:
            if time.time() > deadline:
                break
            
            p = primes[idx]
            old_r = best_residues[idx]
            
            # Try a few random alternatives
            for _ in range(5):
                new_r = rng.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                best_residues[idx] = new_r
                score = _score(best_residues, primes, max_L_check)
                
                if score > best_score:
                    best_score = score
                    improved = True
                    break
                else:
                    best_residues[idx] = old_r
        
        if not improved:
            # Try a different strategy: random restart with a few random changes
            trial = best_residues.copy()
            for _ in range(3):
                idx = rng.randint(0, n - 1)
                p = primes[idx]
                trial[idx] = rng.randint(0, p - 1)
            
            score = _score(trial, primes, max_L_check)
            if score > best_score:
                best_residues = trial
                best_score = score
    
    # Final evaluation with larger max_L
    final_score = _score(best_residues, primes, 100000)
    return best_residues
