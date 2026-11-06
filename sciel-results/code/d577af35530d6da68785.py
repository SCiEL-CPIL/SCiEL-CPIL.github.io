import time
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 145  # safety margin before 150s
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)  # 168
    primes_arr = np.array(primes)
    
    # Initial greedy assignment: r_p = 1 for all (or some pattern)
    # Let's start with r_p = 1 for all primes
    residues = np.ones(n_primes, dtype=np.int64)
    
    def compute_L(res, max_check=5000):
        """Compute the largest L such that every x in [1,L] has coverage >= 2."""
        # For each x in [1, max_check], count coverage
        # coverage[x] = number of primes p where x % p == r_p
        xs = np.arange(1, max_check + 1)
        # Build coverage array
        coverage = np.zeros(max_check + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = res[i]
            # x ≡ r (mod p) means x = r + k*p
            # For x in [1, max_check], find all such x
            if r == 0:
                start = p  # first positive multiple
            else:
                start = r
            if start > max_check:
                continue
            # Mark all x = start, start+p, start+2p, ... <= max_check
            coverage[start::p] += 1
        # Find largest L where coverage[1..L] >= 2
        # Find first x where coverage[x] < 2
        below_two = np.where(coverage[1:] < 2)[0]
        if len(below_two) == 0:
            return max_check
        else:
            return below_two[0]  # x = below_two[0] + 1, so L = below_two[0]
    
    # Initial L
    current_L = compute_L(residues)
    
    # Local search: try changing one residue at a time
    def try_improve(res, current_L, time_limit):
        """Try to improve L by changing one residue."""
        best_L = current_L
        best_res = res.copy()
        
        # Try each prime, and for each, try a few candidate residues
        # Focus on primes that might help the bottleneck
        for i in range(n_primes):
            if time.time() > time_limit:
                break
            p = primes[i]
            old_r = res[i]
            # Try a subset of residues: 0, 1, 2, and maybe some around the bottleneck
            candidates = [0, 1, 2, (old_r + 1) % p, (old_r - 1) % p]
            # Also try residues that would cover the bottleneck x = current_L + 1
            bottleneck_x = current_L + 1
            target_r = bottleneck_x % p
            if target_r not in candidates:
                candidates.append(target_r)
            
            for r in candidates:
                if r == old_r:
                    continue
                res[i] = r
                new_L = compute_L(res)
                res[i] = old_r
                if new_L > best_L:
                    best_L = new_L
                    best_res[i] = r
        
        return best_L, best_res
    
    # Iterative improvement
    for iteration in range(50):
        if time.time() > TIME_LIMIT:
            break
        new_L, new_res = try_improve(residues, current_L, TIME_LIMIT)
        if new_L > current_L:
            current_L = new_L
            residues = new_res
        else:
            # No improvement, try a random perturbation
            if time.time() > TIME_LIMIT:
                break
            # Randomly change a few residues
            for _ in range(5):
                i = np.random.randint(n_primes)
                residues[i] = np.random.randint(primes[i])
            current_L = compute_L(residues)
    
    # Final check
    if time.time() < TIME_LIMIT - 5:
        # One more round of local search
        new_L, new_res = try_improve(residues, current_L, TIME_LIMIT)
        if new_L > current_L:
            current_L = new_L
            residues = new_res
    
    return [int(r) for r in residues]
