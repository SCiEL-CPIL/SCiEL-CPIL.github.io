"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    # Generate primes up to 1000
    # Using a simple sieve for speed
    is_prime = [True] * 1001
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 1001, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n_primes = len(primes)
    
    # We want to maximize L such that 1..L is doubly covered.
    # Let's work on a larger horizon to allow the search to find longer prefixes.
    # If we only look at 1000, we can't see improvements beyond that.
    # 20000 is a reasonable upper bound for search, memory is fine (20000 * 2 bytes).
    MAX_X = 20000
    
    # Initial solution: Greedy construction
    # Strategy: Assign residues to maximize the number of points in [1, MAX_X] 
    # that have coverage >= 2.
    # We process primes in increasing order.
    
    residues = [0] * n_primes
    coverage = np.zeros(MAX_X + 1, dtype=np.int16)
    
    # Helper to update coverage for a prime
    def update_coverage(p, r, delta):
        # x = r, r+p, r+2p...
        # If r == 0, start at p (since x>=1)
        start = r if r != 0 else p
        if start <= MAX_X:
            coverage[start::p] += delta

    # Greedy Phase
    # To make greedy effective for prefix length, we should prioritize covering early numbers.
    # However, standard greedy for "count >= 2" is a good proxy.
    # Let's do a few passes of greedy with different orderings or just one strong pass.
    
    # Pass 1: Smallest primes first (they have the most "leverage" in density)
    for i in range(n_primes):
        p = primes[i]
        best_r = 0
        best_score = -1
        old_r = residues[i]
        
        # Remove old contribution
        update_coverage(p, old_r, -1)
        
        # Try all residues for small p, sample for large p
        if p <= 100:
            candidates = range(p)
        else:
            # For large p, random sampling + neighbors of current best
            candidates = [random.randint(0, p-1) for _ in range(20)]
            candidates.append(old_r)
            # Also try 0 and 1 and p-1 as they are common boundaries
            candidates.extend([0, 1, p-1])
            candidates = list(set(candidates))
            
        for r in candidates:
            if r == old_r and best_score == -1:
                # We already removed old, so if we pick old, we need to re-add it to score?
                # No, we just evaluate.
                pass
            
            update_coverage(p, r, 1)
            # Score: count of x in 1..MAX_X with coverage >= 2
            # Using numpy sum is fast
            score = np.sum(coverage[1:MAX_X+1] >= 2)
            update_coverage(p, r, -1)
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        update_coverage(p, best_r, 1)
        
        if time.time() > deadline - 10:
            break

    # Local Search Phase
    # Objective: Maximize the largest L such that coverage[1..L] >= 2.
    # Current best L
    def get_prefix_length(coverage):
        # Find first index where coverage < 2
        # coverage[0] is unused, we care about 1..MAX_X
        mask = coverage[1:MAX_X+1] >= 2
        if mask.all():
            return MAX_X
        # np.argmax returns index of first True in (mask == False)
        # Invert mask
        first_fail = np.argmax(~mask)
        return int(first_fail)

    current_L = get_prefix_length(coverage)
    best_L = current_L
    best_residues = residues[:]
    
    # Local Search: Hill Climbing
    # Try to improve L by changing one residue at a time.
    # If a change increases L, accept.
    # To avoid getting stuck, we can use Simulated Annealing on L, 
    # but L is discrete and sparse. 
    # Better: "Best Improvement" local search.
    
    no_improve_count = 0
    max_no_improve = 50
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        if no_improve_count > max_no_improve:
            # Try a random restart or perturbation
            # Perturb: change 5 random residues randomly
            for _ in range(5):
                i = rng.randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = rng.randint(0, p - 1)
                update_coverage(p, old_r, -1)
                residues[i] = new_r
                update_coverage(p, new_r, 1)
            current_L = get_prefix_length(coverage)
            no_improve_count = 0
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
            continue

        improved = False
        # Shuffle order to avoid bias
        order = list(range(n_primes))
        rng.shuffle(order)
        
        for i in order:
            p = primes[i]
            old_r = residues[i]
            
            # Generate candidates
            if p <= 50:
                candidates = list(range(p))
            else:
                # Sample candidates
                candidates = [rng.randint(0, p - 1) for _ in range(10)]
                candidates.append(old_r)
                # Add neighbors
                candidates.append((old_r + 1) % p)
                candidates.append((old_r - 1) % p)
                candidates = list(set(candidates))
            
            best_candidate_r = old_r
            best_candidate_L = current_L
            
            update_coverage(p, old_r, -1)
            
            for r in candidates:
                if r == old_r:
                    continue
                update_coverage(p, r, 1)
                new_L = get_prefix_length(coverage)
                update_coverage(p, r, -1)
                
                if new_L > best_candidate_L:
                    best_candidate_L = new_L
                    best_candidate_r = r
            
            # Revert to old state for now, then apply best if improved
            update_coverage(p, old_r, 1)
            
            if best_candidate_r != old_r:
                # Apply the best change
                update_coverage(p, old_r, -1)
                residues[i] = best_candidate_r
                update_coverage(p, best_candidate_r, 1)
                current_L = best_candidate_L
                improved = True
                
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                    
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If we hit 10000 or higher, we are doing very well. 
        # The task says L>=1000 is the target.
        if best_L >= 10000:
             break

    return best_residues
# EVOLVE-BLOCK-END
