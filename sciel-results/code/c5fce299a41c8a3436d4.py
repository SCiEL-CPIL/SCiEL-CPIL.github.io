"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by at least 2 congruences."""
    n = len(residues)
    # For each prime index i, the congruence x = residues[i] (mod primes[i])
    # covers numbers x where x % primes[i] == residues[i]
    
    # We want to find the largest L such that for all x in [1, L], 
    # count of i where x % primes[i] == residues[i] >= 2.
    
    # Efficient approach: for each x from 1 upward, check if it's covered at least twice.
    # But we can't check up to 100000 naively for 168 primes each time (168 * 100000 = 16.8M ops, feasible).
    
    # Let's use a different approach: for each prime i, mark which numbers in [1, max_check] it covers.
    # Then sum up the coverage counts.
    
    # Actually, let's just iterate x from 1 and check coverage.
    # To speed up, we can precompute for each prime the list of x in [1, max_check] it covers.
    
    # Better: use numpy to vectorize.
    # Create array of x from 1 to some limit.
    # For each prime, compute x % p and check if it equals residue.
    
    # Let's try a hybrid: check incrementally.
    
    # First, let's estimate a reasonable upper bound. The seed gives L=84.
    # Let's check up to max_check but stop early if we find a gap.
    
    coverage = np.zeros(max_check + 1, dtype=np.int8)
    for i, (r, p) in enumerate(zip(residues, primes)):
        # Numbers x in [1, max_check] where x % p == r
        # x = r + k*p, for k >= 0, x >= 1
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            xs = np.arange(start, max_check + 1, p)
            coverage[xs] += 1
    
    # Find largest L such that coverage[1..L] >= 2
    # Check from 1 upward
    L = 0
    for x in range(1, max_check + 1):
        if coverage[x] < 2:
            L = x - 1
            break
    else:
        L = max_check
    
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Initial solution from seed
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = _score(best_residues, primes)
    
    # Local search: try to improve by changing one residue at a time
    random.seed(42)
    
    def _local_search(residues, primes, max_iter=5000):
        residues = list(residues)
        current_score = _score(residues, primes)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random prime index
            idx = random.randint(0, len(primes) - 1)
            p = primes[idx]
            old_r = residues[idx]
            
            # Try a few new residues
            improved = False
            for _trial in range(5):
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                residues[idx] = new_r
                new_score = _score(residues, primes)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    residues[idx] = old_r
            
            if not improved:
                # Try to find a better residue for this index by checking all?
                # That's expensive. Let's just continue.
                pass
        
        return residues, current_score
    
    # Run local search
    improved_residues, improved_score = _local_search(best_residues, primes, max_iter=2000)
    if improved_score > best_score:
        best_residues = improved_residues
        best_score = improved_score
    
    # Try a few random restarts
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random initial solution
            random_residues = [random.randint(0, p - 1) for p in primes]
            r_residues, r_score = _local_search(random_residues, primes, max_iter=1000)
            if r_score > best_score:
                best_residues = r_residues
                best_score = r_score
    
    return best_residues
# EVOLVE-BLOCK-END
