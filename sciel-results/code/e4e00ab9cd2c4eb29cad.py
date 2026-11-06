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

def _get_L(counts, max_check=100000):
    """Find the largest L such that counts[1..L] >= 2."""
    # We assume counts[0] is unused.
    # Find first index i >= 1 where counts[i] < 2
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _greedy_construct(primes, order, max_check=100000):
    """
    Greedy set-cover heuristic.
    For each prime in 'order', pick residue to cover the first gap.
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(len(primes), dtype=np.int32)
    current_L = 0
    
    # Map prime value to index in primes list for easy access
    # primes is sorted, so index is just position.
    
    for p in order:
        # Find first gap > current_L
        if current_L < max_check:
            # Search for first x > current_L where counts[x] < 2
            # Optimization: if current_L is 0, start from 1.
            start_search = current_L + 1
            if start_search > max_check:
                g1 = max_check + 1
            else:
                # Find first index in counts[start_search:] where value < 2
                segment = counts[start_search:]
                below_two = np.where(segment < 2)[0]
                if len(below_two) == 0:
                    g1 = max_check + 1
                else:
                    g1 = start_search + int(below_two[0])
        else:
            g1 = max_check + 1
            
        # If g1 > max_check, we are already covering everything.
        # Pick arbitrary residue, e.g., 0.
        if g1 > max_check:
            r = 0
        else:
            # Optimal residue is g1 % p
            r = g1 % p
        
        # Update counts
        residues[primes.index(p)] = r
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
        # Update current_L
        new_L = _get_L(counts, max_check)
        if new_L > current_L:
            current_L = new_L
        else:
            # L should not decrease in greedy add-only phase
            current_L = new_L
            
    return residues, current_L

def _fast_hill_climb(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Fast local search using incremental gap logic.
    """
    n_primes = len(primes)
    improved = True
    passes = 0
    
    while improved and passes < 30:
        passes += 1
        improved = False
        
        for i in range(n_primes):
            if deadline and time.time() > deadline:
                return residues, counts, current_L
                
            p = primes[i]
            current_r = residues[i]
            
            # Build candidate list
            if p <= 30:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                
                # Add residue that covers current first gap
                if current_L < max_check:
                    start_search = current_L + 1
                    segment = counts[start_search:]
                    below_two = np.where(segment < 2)[0]
                    if len(below_two) > 0:
                        g1 = start_search + int(below_two[0])
                        target_r = g1 % p
                        if target_r != current_r:
                            candidates.append(target_r)
                        
                        # Also try residues that cover positions near current_L
                        for offset in [1, 2, 3, 5, 10]:
                            pos = current_L + offset
                            if pos <= max_check:
                                r_off = pos % p
                                if r_off not in candidates:
                                    candidates.append(r_off)
                
                # Add some random candidates
                for _ in range(8):
                    r_rand = np.random.randint(0, p)
                    if r_rand not in candidates:
                        candidates.append(r_rand)
                
                # Cap candidates for large p
                if len(candidates) > 20:
                    candidates = candidates[:20]
                    
            best_r = current_r
            best_L = current_L
            
            for r in candidates:
                if r == current_r:
                    continue
                
                # Apply change
                old_start = current_r if current_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                
                new_start = r if r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                new_L = _get_L(counts, max_check)
                
                if new_L > best_L:
                    best_L = new_L
                    best_r = r
                else:
                    # Revert
                    counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
                    
            # Apply best
            if best_r != current_r:
                residues[i] = best_r
                current_L = best_L
                improved = True
                
    return residues, counts, current_L

def _targeted_gap_fix(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Targeted pass: find the first gap and try to fix it by adjusting one prime's residue.
    """
    if current_L >= max_check:
        return residues, counts, current_L, False
    
    # Find first gap
    start_search = current_L + 1
    segment = counts[start_search:]
    below_two = np.where(segment < 2)[0]
    if len(below_two) == 0:
        return residues, counts, current_L, False
    
    g1 = start_search + int(below_two[0])
    
    # Try to fix by adjusting each prime
    n_primes = len(primes)
    improved_any = False
    
    for i in range(n_primes):
        if deadline and time.time() > deadline:
            return residues, counts, current_L, improved_any
            
        p = primes[i]
        current_r = residues[i]
        
        # The residue that would cover g1
        target_r = g1 % p
        
        if target_r == current_r:
            continue
        
        # Try this change
        old_start = current_r if current_r > 0 else p
        if old_start <= max_check:
            counts[old_start::p] -= 1
        
        new_start = target_r if target_r > 0 else p
        if new_start <= max_check:
            counts[new_start::p] += 1
        
        new_L = _get_L(counts, max_check)
        
        if new_L > current_L:
            residues[i] = target_r
            current_L = new_L
            improved_any = True
            # Re-find gap after improvement
            if current_L >= max_check:
                return residues, counts, current_L, improved_any
            start_search = current_L + 1
            segment = counts[start_search:]
            below_two = np.where(segment < 2)[0]
            if len(below_two) == 0:
                return residues, counts, current_L, improved_any
            g1 = start_search + int(below_two[0])
        else:
            # Revert
            counts[new_start::p] -= 1
            if old_start <= max_check:
                counts[old_start::p] += 1
    
    return residues, counts, current_L, improved_any

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    best_residues = None
    best_score = 0
    
    # Try a few different greedy orders
    orders = [
        primes,  # Increasing
        primes[::-1],  # Decreasing
        list(np.random.permutation(primes))
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
            
        residues, L = _greedy_construct(primes, order, max_check)
        
        # Refine with hill climbing
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] += 1
        
        # First do targeted gap fixing
        residues, counts, L, _ = _targeted_gap_fix(residues, primes, counts, L, max_check, deadline)
        
        # Then refine with hill climbing
        residues, counts, L = _fast_hill_climb(residues, primes, counts, L, max_check, deadline)
        
        # Another round of targeted gap fixing after hill climb
        residues, counts, L, _ = _targeted_gap_fix(residues, primes, counts, L, max_check, deadline)
        
        # Final hill climb pass
        residues, counts, L = _fast_hill_climb(residues, primes, counts, L, max_check, deadline)
        
        if L > best_score:
            best_score = L
            best_residues = residues.copy()
            
    if best_residues is None:
        # Fallback
        best_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
        
    return best_residues.tolist()
# EVOLVE-BLOCK-END
