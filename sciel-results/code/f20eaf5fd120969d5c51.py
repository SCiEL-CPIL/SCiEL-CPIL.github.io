import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill

    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We will search for L up to a reasonable max.
    MAX_CHECK = 50000

    # Precompute indices for each prime and residue
    # indices_cache[i][r] is an array of all x in 1..MAX_CHECK such that x % p_i == r
    indices_cache = [None] * n_primes
    for i, p in enumerate(primes):
        indices_cache[i] = [None] * p
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_CHECK:
                indices_cache[i][r] = np.arange(start, MAX_CHECK + 1, p, dtype=np.int32)
            else:
                indices_cache[i][r] = np.array([], dtype=np.int32)

    def calculate_score(residues):
        """
        Calculate the largest L such that every integer 1..L satisfies at least two congruences.
        """
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            idx = indices_cache[i][r]
            if len(idx) > 0:
                counts[idx] += 1
        
        # Find the first index where count < 2
        # We care about 1..L
        if MAX_CHECK == 0:
            return 0
            
        # Create a boolean array where counts < 2
        fail_mask = counts < 2
        # We only care about indices 1..MAX_CHECK
        # If fail_mask[1:] has any True, find the first one.
        if np.any(fail_mask[1:]):
            first_fail = np.argmax(fail_mask[1:]) + 1
            return first_fail - 1
        else:
            return MAX_CHECK

    def greedy_construct():
        """
        Greedy construction: process primes in increasing order.
        For each prime, select the residue that maximizes the score.
        """
        residues = [0] * n_primes
        
        for i, p in enumerate(primes):
            best_r = 0
            best_gain = -1
            
            # Try all residues for this prime
            for r in range(p):
                residues[i] = r
                # Calculate score
                score = calculate_score(residues)
                if score > best_gain:
                    best_gain = score
                    best_r = r
                # Early stop if we reached MAX_CHECK
                if best_gain == MAX_CHECK:
                    break
            
            residues[i] = best_r
            
            # Check time
            if time.time() - start_time > time_limit - 10:
                break
        
        return residues, best_gain

    best_score = 0
    best_residues = None

    # Phase 1: Greedy Construction
    residues, score = greedy_construct()
    if score > best_score:
        best_score = score
        best_residues = residues[:]
    
    if time.time() - start_time > time_limit - 20:
        return [int(r) for r in best_residues] if best_residues else [0]*n_primes

    # Phase 2: Local Search / Hill Climbing
    # We will polish the greedy result.
    
    def local_search(residues, current_score, time_budget):
        max_no_improve = 0
        no_improve_limit = 100  # Allow more exploration
        
        # Precompute counts for the current state
        counts = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            idx = indices_cache[i][r]
            if len(idx) > 0:
                counts[idx] += 1
                
        while time.time() - start_time < time_budget:
            if max_no_improve > no_improve_limit:
                break
            
            # Find first failure
            fail_mask = counts < 2
            if not np.any(fail_mask[1:]):
                if current_score < MAX_CHECK:
                    current_score = MAX_CHECK
                break
                
            first_fail_idx = np.argmax(fail_mask[1:]) + 1
            x_fail = first_fail_idx
            
            # Try to fix x_fail - try multiple candidate moves
            move_made = False
            improved = False
            
            # Strategy: 
            # 1. Try to cover x_fail by changing a prime's residue to x_fail % p.
            #    Prefer primes that currently cover x_fail with count 1 or 0.
            #    Actually, we want to increase coverage at x_fail.
            #    Currently counts[x_fail] < 2.
            #    We need to change a residue r_i such that x_fail % p_i == new_r_i.
            
            # Identify primes that currently do NOT cover x_fail
            # And primes that DO cover x_fail (count 1)
            
            candidates = []
            for i in range(n_primes):
                p = primes[i]
                target_r = x_fail % p
                current_r = residues[i]
                
                if target_r == current_r:
                    # This prime already covers x_fail
                    continue
                
                # If we change residue to target_r, we add 1 to x_fail.
                # We must ensure we don't break the prefix [1, x_fail-1].
                # Specifically, for any y in [1, x_fail-1] where y % p == current_r,
                # counts[y] must be > 2 (so that removing 1 keeps it >= 2).
                
                # Get indices for current_r
                old_indices = indices_cache[i][current_r]
                # Filter indices < x_fail
                if len(old_indices) == 0:
                    cutoff_idx = 0
                else:
                    cutoff_idx = np.searchsorted(old_indices, x_fail, side='left')
                
                if cutoff_idx == 0:
                    # No affected indices in the critical prefix
                    valid = True
                else:
                    affected_counts = counts[old_indices[:cutoff_idx]]
                    if np.any(affected_counts == 2):
                        valid = False
                    else:
                        valid = True
                
                if valid:
                    # Calculate potential gain
                    # The gain is roughly (new_score - current_score)
                    # But we can estimate: does this move allow us to extend L?
                    # If counts[x_fail] becomes 2, and all subsequent counts are >= 2, L extends.
                    # Simple heuristic: prefer moves that fix x_fail.
                    # Since all valid moves fix x_fail (by definition of target_r),
                    # we can pick one that minimizes "risk" or is random.
                    # Let's pick the one with the smallest prime (affects more numbers)
                    # or just randomize to escape local minima.
                    candidates.append(i)
            
            if candidates:
                # Pick a candidate
                # Prefer smaller primes? Or random?
                # Randomizing helps escape local minima.
                # Let's try a few random candidates from the list.
                np.random.shuffle(candidates)
                
                for i in candidates[:5]: # Try up to 5 candidates
                    p = primes[i]
                    old_r = residues[i]
                    new_r = x_fail % p
                    
                    # Apply move
                    residues[i] = new_r
                    
                    # Update counts incrementally?
                    # For simplicity and correctness, recalculate score.
                    # But we can update counts array to speed up next iterations.
                    old_indices = indices_cache[i][old_r]
                    new_indices = indices_cache[i][new_r]
                    
                    counts[old_indices] -= 1
                    counts[new_indices] += 1
                    
                    # Calculate new score
                    new_score = calculate_score(residues)
                    
                    if new_score > current_score:
                        current_score = new_score
                        move_made = True
                        improved = True
                        max_no_improve = 0
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                        counts[old_indices] -= 1
                        counts[new_indices] += 1
                        
                        # If we didn't improve, maybe try a neutral move?
                        # For now, skip.
                        pass
            
            if not move_made:
                # Stuck. Try random perturbation
                escaped = False
                for _ in range(20):
                    if time.time() - start_time > time_budget:
                        break
                    i = int(np.random.randint(0, n_primes))
                    p = primes[i]
                    old_r = residues[i]
                    new_r = int(np.random.randint(0, p))
                    if new_r == old_r:
                        continue
                    
                    # Apply move
                    residues[i] = new_r
                    old_indices = indices_cache[i][old_r]
                    new_indices = indices_cache[i][new_r]
                    
                    counts[old_indices] -= 1
                    counts[new_indices] += 1
                    
                    new_score = calculate_score(residues)
                    
                    if new_score > current_score:
                        current_score = new_score
                        escaped = True
                        max_no_improve = 0
                        break
                    elif new_score == current_score and max_no_improve > 20:
                        # Accept neutral move to escape local minima
                        escaped = True
                        max_no_improve += 1
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                        counts[old_indices] -= 1
                        counts[new_indices] += 1
                        
                if not escaped:
                    max_no_improve += 1
                    if max_no_improve > no_improve_limit:
                        break
        
        return residues, current_score

    # Run local search on the greedy result
    residues, score = local_search(residues[:], best_score, time_limit - 5)
    if score > best_score:
        best_score = score
        best_residues = residues[:]
    
    # If we have time, try one more restart with random init + local search
    if time.time() - start_time < time_limit - 20:
        for restart in range(10):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Random initialization
            residues = [int(np.random.randint(0, p)) for p in primes]
            current_score = calculate_score(residues)
            
            # Local search
            residues, score = local_search(residues, current_score, time_limit - 5)
            
            if score > best_score:
                best_score = score
                best_residues = residues[:]
    
    # Return the best residues found
    if best_residues is None:
        # Fallback
        return [int(((i + 2) // 2) % p) for i, p in enumerate(primes)]
        
    return [int(r) for r in best_residues]
