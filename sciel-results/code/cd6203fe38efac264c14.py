"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    """Generate all primes up to n using sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered at least twice."""
    n = len(primes)
    # For each prime p with residue r, the integers covered in [1, max_L] are:
    # r, r+p, r+2p, ... up to max_L
    # We need count[x] >= 2 for all x in 1..L
    
    # Use an efficient approach: for each x, count how many primes cover it
    # But max_L can be large, so we need to be smart.
    
    # Alternative: compute coverage incrementally
    # For a given L, check if all 1..L have count >= 2
    
    # Let's use a different approach: 
    # For each prime p with residue r, it covers x where x ≡ r (mod p)
    # i.e., x = r + kp for k >= 0 (and x >= 1)
    
    # The uncovered integers are those with count < 2.
    # We want the smallest x >= 1 with count < 2, then L = x - 1.
    
    # Efficient method: use the fact that we only need to check up to some bound.
    # If L >= 1000, we're done for the task's main goal.
    
    # Let's compute counts for x = 1, 2, 3, ... until we find a gap.
    # But this could be slow if L is large.
    
    # Better: For each x, the number of covering primes is the number of (p, r) 
    # such that x ≡ r (mod p).
    
    # Let's use numpy for speed: create an array and add contributions.
    
    # Actually, let's think differently. The maximum useful L is bounded.
    # With 168 primes, each covering ~L/p integers in [1, L], total coverage ~L * sum(1/p) 
    # ≈ L * log(log(1000)) ≈ L * 2.6 or so. For double coverage, need ~2L total,
    # so L * 2.6 >= 2L is always satisfied in terms of total, but distribution matters.
    
    # Let's just compute directly but cap at a reasonable value.
    # We'll compute counts for x = 1 to min(max_L, some_upper_bound)
    
    # Upper bound: if every prime covers at least one integer in [1, L], 
    # the minimum L where a prime might not cover anything is L < p.
    # For p <= 1000, if L >= 1000, every prime covers at least one integer in [1, L].
    # So for L >= 1000, we can check efficiently.
    
    # Let's compute counts up to max_check = min(max_L, 5000) first, 
    # and if all covered, try larger.
    
    max_check = min(max_L, 10000)
    
    # counts[x] = number of primes p where x ≡ r_p (mod p)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for p, r in zip(primes, residues):
        if r == 0:
            # covers p, 2p, 3p, ...
            start = p
        else:
            start = r
        for x in range(start, max_check + 1, p):
            counts[x] += 1
    
    # Find the smallest x >= 1 with counts[x] < 2
    L = 0
    for x in range(1, max_check + 1):
        if counts[x] < 2:
            L = x - 1
            break
        L = x
    
    return L

def _local_search(residues, primes, initial_L, time_limit=120):
    """Improve residues via local search to maximize L."""
    start_time = time.time()
    n = len(primes)
    best_L = initial_L
    best_residues = residues.copy()
    
    def compute_L_fast(res):
        """Fast computation of L for a given residue set."""
        # We only need to check up to best_L + some margin
        # But for correctness, we need to find the actual L.
        # Optimization: if we know previous L, we can start checking from there.
        # But changing one residue can affect many positions.
        
        # For now, compute up to a reasonable bound.
        max_check = min(100000, best_L + 500)
        
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for p, r in zip(primes, res):
            if r == 0:
                start = p
            else:
                start = r
            for x in range(start, max_check + 1, p):
                counts[x] += 1
        
        L = 0
        for x in range(1, max_check + 1):
            if counts[x] < 2:
                L = x - 1
                break
            L = x
        return L
    
    # Local search: try changing one residue at a time
    # Use simulated annealing-like approach with restarts
    
    current_L = best_L
    current_residues = best_residues.copy()
    
    # Precompute for speed: for each prime index i, and each possible new residue r,
    # what's the effect on coverage?
    # This is complex, so let's just do direct evaluation but with early termination.
    
    def compute_L_early(res, upper_bound):
        """Compute L with early termination."""
        max_check = min(upper_bound, 100000)
        
        # Quick check: compute counts only where needed
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for p, r in zip(primes, res):
            if r == 0:
                start = p
            else:
                start = r
            for x in range(start, max_check + 1, p):
                counts[x] += 1
        
        L = 0
        for x in range(1, max_check + 1):
            if counts[x] < 2:
                L = x - 1
                break
            L = x
        return L
    
    # Greedy hill climbing with random restarts
    max_iters = 5000
    iter_count = 0
    
    while iter_count < max_iters:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:  # Safety margin
            break
        
        # Try to improve by changing one residue
        improved = False
        # Randomly select which prime to modify
        i = random.randint(0, n - 1)
        p = primes[i]
        
        # Try a few random new residues
        for _ in range(3):
            new_r = random.randint(0, p - 1)
            if new_r == current_residues[i]:
                continue
            
            old_r = current_residues[i]
            current_residues[i] = new_r
            
            # Compute new L
            new_L = compute_L_early(current_residues, best_L + 200)
            
            if new_L > current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = current_residues.copy()
                improved = True
                break
            else:
                # Revert
                current_residues[i] = old_r
        
        if not improved:
            # Try a kick: change 2-3 residues randomly
            if random.random() < 0.3:
                for _ in range(random.randint(2, 3)):
                    i2 = random.randint(0, n - 1)
                    p2 = primes[i2]
                    current_residues[i2] = random.randint(0, p2 - 1)
                current_L = compute_L_early(current_residues, best_L + 500)
                if current_L < best_L - 50:
                    # Too bad, revert to best
                    current_residues = best_residues.copy()
                    current_L = best_L
        
        iter_count += 1
    
    return best_residues, best_L

def construct():
    """Construct 168 residues, one for each prime p <= 1000 in increasing order."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    primes = _primes_up_to(1000)
    n = len(primes)
    
    # Initial construction: the seed approach
    # The i-th prime covers the integer ceil((i+1)/2) roughly
    # Better initial: use a more structured approach
    
    # Strategy: for each prime p, choose r_p such that p's coverage is "spread out"
    # to help cover gaps.
    
    # Initial guess: r_p = 0 for all p (covers multiples of p)
    # Or: r_p = 1 for all p (covers 1, 1+p, 1+2p, ...)
    
    # Let's start with a randomized initial solution and then local search.
    
    # Multiple random restarts with local search
    best_residues = None
    best_L = 0
    
    # First, try the seed approach
    residues_seed = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    L_seed = _compute_L(residues_seed, primes)
    best_L = L_seed
    best_residues = residues_seed.copy()
    
    # Try a few structured initializations
    # r_p = 1 for all
    residues_ones = [1 % p for p in primes]
    L_ones = _compute_L(residues_ones, primes)
    if L_ones > best_L:
        best_L = L_ones
        best_residues = residues_ones.copy()
    
    # r_p = p-1 for all (covers -1 mod p, i.e., p-1, 2p-1, ...)
    residues_m1 = [(p - 1) % p for p in primes]
    L_m1 = _compute_L(residues_m1, primes)
    if L_m1 > best_L:
        best_L = L_m1
        best_residues = residues_m1.copy()
    
    # Now do local search from the best initial
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        best_residues, best_L = _local_search(
            best_residues.copy(), primes, best_L, time_limit=deadline - start_time
        )
    
    # Final verification
    final_L = _compute_L(best_residues, primes)
    if final_L < best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
