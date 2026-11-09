import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    N = 1000
    x = np.arange(1, N + 1, dtype=np.int32)
    
    # Precompute masks for each prime: mask[i] is a list of booleans for x % p == r
    # Actually, to save memory and time, we compute (x % p) on the fly for the specific p
    # But since we do many evaluations, precomputing x % p for all p is useful.
    # x_mod_p[i] = x % primes[i]
    x_mod_p = np.zeros((num_primes, N), dtype=np.int32)
    for i, p in enumerate(primes):
        x_mod_p[i] = x % p
        
    def compute_counts(residues):
        counts = np.zeros(N, dtype=np.int32)
        for i in range(num_primes):
            r = residues[i]
            counts += (x_mod_p[i] == r)
        return counts
        
    def compute_L(counts):
        # Find the first index where count < 2
        # We want the largest L such that counts[0..L-1] >= 2
        # So L is the index of the first failure, or N if no failure
        fail_mask = counts < 2
        if np.any(fail_mask):
            return np.argmax(fail_mask)
        return N

    # Greedy Initialization
    # Start with all residues 0
    residues = np.zeros(num_primes, dtype=np.int32)
    counts = compute_counts(residues)
    
    # Greedily improve: for each prime, pick the residue that maximizes the minimum count in the "bottleneck" region
    # Or simply: iterate primes, for each prime pick the residue that maximizes the score L
    # This is O(P * P * N) which is 168 * 168 * 1000 ~ 28M ops, acceptable.
    
    # A faster greedy: 
    # 1. Initialize residues randomly.
    # 2. For each prime i, try all possible residues r. 
    #    Calculate the new L. Keep the r that gives the max L.
    # This is O(P * P * N) per pass.
    
    residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int32)
    counts = compute_counts(residues)
    
    # Greedy Pass 1: Single moves
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(num_primes):
            p = primes[i]
            best_r = residues[i]
            best_L = compute_L(counts)
            
            old_r = residues[i]
            # Try all residues for this prime
            for r in range(p):
                if r == old_r:
                    continue
                # Update counts
                counts -= (x_mod_p[i] == old_r)
                counts += (x_mod_p[i] == r)
                new_L = compute_L(counts)
                
                if new_L > best_L:
                    best_L = new_L
                    best_r = r
            
            # Apply best
            if best_r != old_r:
                counts -= (x_mod_p[i] == old_r)
                counts += (x_mod_p[i] == best_r)
                residues[i] = best_r
                improved = True

    current_L = compute_L(counts)
    best_L = current_L
    best_residues = residues.copy()
    
    # Iterated Local Search
    # We will perform local search (single and pair moves) and then perturb.
    
    def local_search(residues, counts, current_L, max_iterations=5000):
        # Returns improved residues, counts, L
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and time.time() < deadline - 2:
            improved = False
            iter_count += 1
            
            # Try single moves
            for _ in range(num_primes * 2): # More attempts than primes to ensure coverage
                i = random.randint(0, num_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                counts -= (x_mod_p[i] == old_r)
                counts += (x_mod_p[i] == new_r)
                new_L = compute_L(counts)
                
                if new_L > current_L:
                    residues[i] = new_r
                    current_L = new_L
                    improved = True
                else:
                    counts -= (x_mod_p[i] == new_r)
                    counts += (x_mod_p[i] == old_r)
            
            # Try pair moves
            for _ in range(num_primes): # Fewer pair moves as they are more expensive
                i = random.randint(0, num_primes - 1)
                j = random.randint(0, num_primes - 1)
                if i == j:
                    continue
                    
                p_i = primes[i]
                p_j = primes[j]
                old_r_i = residues[i]
                old_r_j = residues[j]
                new_r_i = random.randint(0, p_i - 1)
                new_r_j = random.randint(0, p_j - 1)
                
                if new_r_i == old_r_i and new_r_j == old_r_j:
                    continue
                
                # Apply changes
                counts -= (x_mod_p[i] == old_r_i)
                counts += (x_mod_p[i] == new_r_i)
                counts -= (x_mod_p[j] == old_r_j)
                counts += (x_mod_p[j] == new_r_j)
                
                new_L = compute_L(counts)
                
                if new_L > current_L:
                    residues[i] = new_r_i
                    residues[j] = new_r_j
                    current_L = new_L
                    improved = True
                else:
                    # Revert
                    counts -= (x_mod_p[i] == new_r_i)
                    counts += (x_mod_p[i] == old_r_i)
                    counts -= (x_mod_p[j] == new_r_j)
                    counts += (x_mod_p[j] == old_r_j)
                    
        return residues, counts, current_L

    # Main ILS loop
    while time.time() < deadline - 5:
        # Run local search
        residues, counts, current_L = local_search(residues, counts, current_L)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        # If we reached the maximum possible L (1000), we can stop
        if best_L >= 1000:
            break
            
        # Perturb: change a few random residues
        num_perturb = random.randint(3, 10)
        for _ in range(num_perturb):
            i = random.randint(0, num_primes - 1)
            p = primes[i]
            residues[i] = random.randint(0, p - 1)
            
        # Recompute counts
        counts = compute_counts(residues)
        current_L = compute_L(counts)
        
    return [int(r) for r in best_residues]
