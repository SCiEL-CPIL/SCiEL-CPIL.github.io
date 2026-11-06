"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]

def _compute_counts(residues, primes, max_check=100000):
    """Compute coverage counts for all integers 1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _fast_score_from_counts(counts, max_check=100000):
    """Find first index where count < 2, starting from 1. Returns 0-indexed offset."""
    # Use a sliding window approach: find first position with count < 2
    # Instead of scanning the whole array, we can use numpy's argmax on a boolean
    below = counts[1:] < 2
    if not below.any():
        return max_check
    return int(np.argmax(below))

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    max_check = 100000
    
    # Deadline guard: 150s wall-clock limit, use 145s for safety
    deadline = time.time() + 145
    
    # Start with the seed solution
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = _score(best_residues, primes)
    
    # Phase 1: Greedy hill-climbing with incremental updates
    # Maintain counts array and update incrementally when changing a residue
    improved = True
    round_num = 0
    max_rounds = 500
    
    while improved and round_num < max_rounds and time.time() < deadline:
        improved = False
        round_num += 1
        
        # Recompute counts from current best
        counts = _compute_counts(best_residues, primes, max_check)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # Generate candidates: neighbors, strategic positions, and randoms
            candidates = set()
            
            # Nearby residues
            for delta in range(-8, 9):
                new_r = (current_r + delta) % p
                candidates.add(new_r)
            
            # Strategic: try residues that cover small integers
            for target in range(1, min(30, p)):
                candidates.add(target % p)
            
            # Random candidates for larger primes
            if p > 15:
                for _ in range(12):
                    candidates.add(int(np.random.randint(0, p)))
            
            best_local_r = current_r
            best_local_score = best_score
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                
                # Update counts incrementally
                # Remove old coverage
                old_start = current_r if current_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                
                # Add new coverage
                new_start = new_r if new_r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                # Compute score from updated counts using fast method
                s = _fast_score_from_counts(counts, max_check)
                
                if s > best_local_score:
                    best_local_score = s
                    best_local_r = new_r
                else:
                    # Revert
                    if new_start <= max_check:
                        counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
            
            if best_local_r != current_r:
                best_residues[i] = best_local_r
                best_score = best_local_score
                improved = True
    
    # Phase 2: Multi-residue moves (swap or change 2 residues at once)
    # This can escape local optima
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        counts = _compute_counts(best_residues, primes, max_check)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Pick two random positions to modify
            i1, i2 = np.random.choice(n, size=2, replace=False)
            p1, p2 = primes[i1], primes[i2]
            r1, r2 = best_residues[i1], best_residues[i2]
            
            # Try random new values for both
            for _ in range(15):
                new_r1 = int(np.random.randint(0, p1))
                new_r2 = int(np.random.randint(0, p2))
                
                if new_r1 == r1 and new_r2 == r2:
                    continue
                
                # Apply changes
                old_start1 = r1 if r1 > 0 else p1
                old_start2 = r2 if r2 > 0 else p2
                new_start1 = new_r1 if new_r1 > 0 else p1
                new_start2 = new_r2 if new_r2 > 0 else p2
                
                if old_start1 <= max_check:
                    counts[old_start1::p1] -= 1
                if old_start2 <= max_check:
                    counts[old_start2::p2] -= 1
                if new_start1 <= max_check:
                    counts[new_start1::p1] += 1
                if new_start2 <= max_check:
                    counts[new_start2::p2] += 1
                
                s = _fast_score_from_counts(counts, max_check)
                
                if s > best_score:
                    best_residues[i1] = new_r1
                    best_residues[i2] = new_r2
                    best_score = s
                    break
                else:
                    # Revert
                    if new_start1 <= max_check:
                        counts[new_start1::p1] -= 1
                    if new_start2 <= max_check:
                        counts[new_start2::p2] -= 1
                    if old_start1 <= max_check:
                        counts[old_start1::p1] += 1
                    if old_start2 <= max_check:
                        counts[old_start2::p2] += 1
    
    # Phase 3: Fine-tuning pass with smaller steps near the boundary
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        counts = _compute_counts(best_residues, primes, max_check)
        # Focus on residues that affect the boundary region
        boundary = best_score
        for i in range(n):
            if time.time() > deadline:
                break
            p = primes[i]
            current_r = best_residues[i]
            
            # Try fine adjustments
            for delta in range(-2, 3):
                new_r = (current_r + delta) % p
                if new_r == current_r:
                    continue
                
                old_start = current_r if current_r > 0 else p
                new_start = new_r if new_r > 0 else p
                
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                s = _fast_score_from_counts(counts, max_check)
                
                if s > best_score:
                    best_residues[i] = new_r
                    best_score = s
                    current_r = new_r
                    old_start = new_start
                else:
                    if new_start <= max_check:
                        counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
    
    # Final verification
    final_score = _score(best_residues, primes)
    return best_residues
# EVOLVE-BLOCK-END
