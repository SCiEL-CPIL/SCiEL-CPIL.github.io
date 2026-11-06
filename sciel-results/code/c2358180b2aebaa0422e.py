import time
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _first_start(r, p):
    """First x >= 1 such that x ≡ r (mod p)"""
    if r == 0:
        return p
    return r

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Target window size for optimization
    # We want L >= 1000. Let's optimize on a window of 5000 to be safe.
    # If we cover 5000, we definitely cover 1000.
    M = 5000
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initialize residues to 0
    residues = [0] * n
    counts = np.zeros(M + 1, dtype=np.int32)
    
    def _update_counts(counts, old_r, new_r, p, max_check):
        old_start = _first_start(old_r, p)
        if old_start <= max_check:
            counts[old_start::p] -= 1
        new_start = _first_start(new_r, p)
        if new_start <= max_check:
            counts[new_start::p] += 1

    def _find_L(counts):
        # Find first x in 1..len(counts)-1 where counts[x] < 2
        # If all are >= 2, return len(counts)-1
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return len(counts) - 1
        return below[0]

    # --- Phase 1: Greedy Construction ---
    # Process primes in increasing order.
    # For each prime, pick the residue that maximizes the number of 
    # "single-covered" integers (count == 1) that become "double-covered" (count == 2).
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        best_score2 = -1 # Secondary score: count of 0s covered
        
        for r in range(p):
            # Calculate gain for this residue
            # We need to count how many x in [1, M] with x % p == r have counts[x] == 1
            # and how many have counts[x] == 0
            
            start = _first_start(r, p)
            if start > M:
                # No x in range
                score1 = 0
                score2 = 0
            else:
                # Vectorized counting
                # Indices: start, start+p, start+2p, ... <= M
                indices = np.arange(start, M + 1, p)
                c = counts[indices]
                score1 = np.sum(c == 1)
                score2 = np.sum(c == 0)
            
            # We prioritize fixing single covers (score1)
            # If score1 is the same, we prefer covering more zeros (score2) to build up coverage
            if score1 > best_score or (score1 == best_score and score2 > best_score2):
                best_score = score1
                best_score2 = score2
                best_r = r
        
        # Apply best residue
        residues[i] = best_r
        _update_counts(counts, 0, best_r, p, M)
        
        # Time check
        if time.time() > deadline:
            break

    # --- Phase 2: Local Search Refinement ---
    # Now we have a decent initial configuration.
    # We try to extend L by fixing the first gap.
    
    current_L = _find_L(counts)
    
    # If we hit the boundary M, we might want to extend M, but for now we work within M.
    # If L < M, we try to improve it.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        if current_L >= M:
            # We have covered the entire window. 
            # We could increase M, but let's just return if we are well above 1000.
            # If we want to push higher, we can double M and re-optimize, but that's complex.
            # Given the target is 1000, and M=5000, this is likely sufficient.
            break
            
        target_x = current_L + 1
        
        # Find the best single-prime change that fixes target_x or improves L
        best_candidate_score = current_L
        best_candidate_p_idx = -1
        best_candidate_new_r = -1
        best_candidate_old_r = -1
        
        # Iterate through primes
        # Optimization: Only consider primes that can potentially help.
        # Changing p changes x % p. 
        # If we set r_p = target_x % p, then target_x becomes covered by p.
        # We need to see if this increases the minimum count in the prefix.
        
        for i, p in enumerate(primes):
            old_r = residues[i]
            new_r = target_x % p
            
            if new_r == old_r:
                continue
            
            # Apply change
            _update_counts(counts, old_r, new_r, p, M)
            
            # Evaluate new L
            new_L = _find_L(counts)
            
            if new_L > best_candidate_score:
                best_candidate_score = new_L
                best_candidate_p_idx = i
                best_candidate_new_r = new_r
                best_candidate_old_r = old_r
            
            # Revert
            _update_counts(counts, new_r, old_r, p, M)
        
        if best_candidate_p_idx != -1:
            # Apply the best change
            _update_counts(counts, best_candidate_old_r, best_candidate_new_r, primes[best_candidate_p_idx], M)
            residues[best_candidate_p_idx] = best_candidate_new_r
            current_L = best_candidate_score
            no_improve_count = 0
        else:
            # No improvement by fixing the immediate next gap with a single prime change.
            # Try random perturbations to escape local minima.
            no_improve_count += 1
            
            if no_improve_count > 100:
                # Random restart of a few residues
                # Pick a random prime and a random residue
                # Accept if it doesn't reduce L
                improved = False
                for _ in range(10):
                    i = np.random.randint(0, n)
                    old_r = residues[i]
                    new_r = np.random.randint(0, primes[i])
                    
                    _update_counts(counts, old_r, new_r, primes[i], M)
                    new_L = _find_L(counts)
                    
                    if new_L >= current_L:
                        residues[i] = new_r
                        if new_L > current_L:
                            current_L = new_L
                            improved = True
                    else:
                        _update_counts(counts, new_r, old_r, primes[i], M)
                
                if improved:
                    no_improve_count = 0
                else:
                    # If stuck for a long time, try a more aggressive random restart
                    if no_improve_count > 1000:
                        # Reset a random subset of primes to random values
                        for _ in range(5):
                            i = np.random.randint(0, n)
                            old_r = residues[i]
                            new_r = np.random.randint(0, primes[i])
                            _update_counts(counts, old_r, new_r, primes[i], M)
                            residues[i] = new_r
                        current_L = _find_L(counts)
                        no_improve_count = 0

    return residues

