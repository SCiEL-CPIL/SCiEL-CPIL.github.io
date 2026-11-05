"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time

def construct():
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    
    # Dynamic window management
    # We start with a window and expand it if fully covered
    initial_window = 1000
    max_window = 100000 # Cap for memory/time
    
    residues = [0] * n_primes
    
    # Helper to compute counts for a given window size
    def compute_counts(residues, primes, window_size):
        counts = np.zeros(window_size + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r
            if start == 0:
                start = p
            if start <= window_size:
                counts[start::p] += 1
        return counts

    def find_first_gap(counts, window_size):
        """Returns the first index x in 1..window_size where counts[x] < 2, or window_size if none."""
        # We want the first x >= 1 such that counts[x] < 2
        # If all counts[1:window_size+1] >= 2, return window_size
        # Note: The problem requires 1..L to be covered.
        # If counts[1] < 2, L=0.
        # If counts[1]>=2, counts[2]<2, L=1.
        
        # Efficiently find first index where count < 2
        # We look at counts[1:window_size+1]
        if window_size == 0:
            return 0
            
        bad_indices = np.where(counts[1:window_size+1] < 2)[0]
        if len(bad_indices) == 0:
            return window_size
        else:
            # bad_indices[0] is relative to slice starting at 1
            # So actual index is bad_indices[0] + 1
            # L is the number of integers 1..L covered.
            # If gap is at x, L = x - 1.
            return bad_indices[0]

    # Greedy Bootstrap
    # Start with random residues
    random.seed(42)
    residues = [random.randint(0, p - 1) for p in primes]
    
    # We will use a greedy phase to quickly find a good starting point
    # Strategy: Look at the first gap. Try to fix it by adjusting one prime.
    
    window_size = initial_window
    counts = compute_counts(residues, primes, window_size)
    current_L = find_first_gap(counts, window_size)
    best_L = current_L
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Phase 1: Greedy Improvement
    # We try to extend L by fixing the first gap.
    for _ in range(5000): # Limit greedy steps
        if time.time() - start_time > time_limit:
            break
            
        if current_L >= window_size:
            # Expand window
            new_window = min(window_size * 2, max_window)
            if new_window == window_size:
                break
            # Recalculate counts for new window
            # We can just recompute or extend. Recomputing is simpler for now.
            counts = compute_counts(residues, primes, new_window)
            window_size = new_window
            current_L = find_first_gap(counts, window_size)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
            continue
            
        # Find first gap x
        gap_x = current_L + 1
        
        # Try to fix gap_x by changing one prime's residue
        # For each prime p, the required residue is r = gap_x % p
        # We evaluate the "quality" of this move.
        # Quality: How many points in [1, window_size] does this improve?
        # Specifically, we want to avoid breaking coverage in [1, current_L].
        
        best_prime_idx = -1
        best_new_L = current_L
        best_new_r = -1
        
        # To speed up, we can sample primes or just iterate all 168 (which is fast)
        for i, p in enumerate(primes):
            new_r = gap_x % p
            old_r = residues[i]
            
            if new_r == old_r:
                continue
                
            # Calculate impact
            # We need to see if changing r_i to new_r increases L.
            # This requires updating counts.
            # Instead of full update, we can estimate or do a fast check.
            # Given 168 primes, doing 168 full count updates per step might be slow if we do many steps.
            # But 168 * (window_size/p) operations.
            # Let's just do the update for the candidate and revert if worse.
            
            # Save old state
            old_counts_segment = counts.copy() # This is expensive. 
            # Better: Update in place, check, revert.
            
            start_old = old_r
            if start_old == 0: start_old = p
            if start_old <= window_size:
                counts[start_old::p] -= 1
            
            start_new = new_r
            if start_new == 0: start_new = p
            if start_new <= window_size:
                counts[start_new::p] += 1
            
            new_L = find_first_gap(counts, window_size)
            
            # Revert
            if start_new <= window_size:
                counts[start_new::p] -= 1
            if start_old <= window_size:
                counts[start_old::p] += 1
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_prime_idx = i
                best_new_r = new_r
        
        if best_prime_idx != -1:
            # Apply best move
            i = best_prime_idx
            p = primes[i]
            old_r = residues[i]
            new_r = best_new_r
            
            start_old = old_r
            if start_old == 0: start_old = p
            if start_old <= window_size:
                counts[start_old::p] -= 1
            
            start_new = new_r
            if start_new == 0: start_new = p
            if start_new <= window_size:
                counts[start_new::p] += 1
            
            residues[i] = new_r
            current_L = best_new_L
            
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
        else:
            # No single prime move improves L. Break greedy phase.
            break

    # Phase 2: Simulated Annealing with Targeted Moves
    # Start from best_residues
    residues = best_residues[:]
    counts = compute_counts(residues, primes, window_size)
    current_L = find_first_gap(counts, window_size)
    
    T = 10.0
    cooling_rate = 0.999
    T_min = 0.01
    
    while time.time() - start_time < time_limit:
        if current_L >= window_size:
            # Expand window
            new_window = min(window_size * 2, max_window)
            if new_window == window_size:
                # If we can't expand, keep optimizing or stop
                # If we are at max_window, just keep optimizing
                pass
            else:
                counts = compute_counts(residues, primes, new_window)
                window_size = new_window
                current_L = find_first_gap(counts, window_size)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                # Reset T? Or keep it.
                T = max(T, 5.0)
                continue

        gap_x = current_L + 1
        
        # Move Selection
        # 80% chance: Targeted Move (fix gap_x)
        # 20% chance: Random Move
        if random.random() < 0.8:
            # Pick a random prime
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            new_r = gap_x % p
        else:
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            new_r = random.randint(0, p - 1)
            
        old_r = residues[idx]
        if new_r == old_r:
            continue
            
        # Apply move
        start_old = old_r
        if start_old == 0: start_old = p
        if start_old <= window_size:
            counts[start_old::p] -= 1
            
        start_new = new_r
        if start_new == 0: start_new = p
        if start_new <= window_size:
            counts[start_new::p] += 1
            
        new_L = find_first_gap(counts, window_size)
        
        # Acceptance
        if new_L >= current_L:
            current_L = new_L
            residues[idx] = new_r
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
        else:
            delta = current_L - new_L
            if delta < T:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
                
            if random.random() < prob:
                current_L = new_L
                residues[idx] = new_r
            else:
                # Revert
                if start_new <= window_size:
                    counts[start_new::p] -= 1
                if start_old <= window_size:
                    counts[start_old::p] += 1
                # residues[idx] remains old_r
        
        T *= cooling_rate
        if T < T_min:
            # Restart temperature
            T = 5.0

    return best_residues

# EVOLVE-BLOCK-END
