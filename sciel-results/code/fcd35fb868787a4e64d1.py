import time
import random
import numpy as np
from collections import defaultdict

def _primes_upto(n):
    """Generate primes up to n using sieve."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    # Safety margin: 140s hard limit, 150s kill
    deadline = start_time + 140
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    primes_np = np.array(primes, dtype=np.int64)
    
    # Precompute coverage indices for each prime and residue
    # cover_indices[i] is a list of arrays? No, better:
    # For prime i, residue r, the indices are np.arange(start, MAX_L+1, p)
    # We can precompute the start offsets.
    
    # We need a fast way to compute coverage and score.
    # Score: largest L such that cov[1..L] >= 2.
    # cov is an array of length MAX_L.
    
    MAX_L = 5000  # Search space for L. If we find 1000, we might push higher.
    
    # Precompute for each prime i, the array of indices for each residue?
    # That's 168 * p_i arrays. Total elements ~ sum(p_i * MAX_L/p_i) = 168 * MAX_L ~ 840k.
    # This is feasible and makes updates O(1) (just adding/subtracting arrays).
    
    # cover_arrays[i] is a list of arrays, cover_arrays[i][r] contains indices x in [1, MAX_L] s.t. x % p_i == r.
    # Note: if r==0, indices are p, 2p, ...
    
    cover_arrays = []
    for i, p in enumerate(primes):
        res_arrays = []
        for r in range(p):
            # x = r + k*p
            # x >= 1
            # If r == 0, first x is p (since x=0 is not in 1..L)
            # If r > 0, first x is r (if r <= MAX_L)
            if r == 0:
                start = p
            else:
                start = r
            
            if start <= MAX_L:
                arr = np.arange(start, MAX_L + 1, p)
            else:
                arr = np.array([], dtype=np.int64)
            res_arrays.append(arr)
        cover_arrays.append(res_arrays)
    
    def compute_cov(residues):
        """
        residues: list or array of length n_primes
        Returns: np.array cov of length MAX_L + 1 (1-indexed effectively, index 0 unused)
        """
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i in range(n_primes):
            r = residues[i]
            arr = cover_arrays[i][r]
            if len(arr) > 0:
                cov[arr] += 1
        return cov
    
    def get_score(cov):
        """
        cov: np.array
        Returns: largest L such that cov[1..L] >= 2
        """
        # We want the first index where cov < 2
        # cov[0] is 0, we care about 1..MAX_L
        # valid = cov[1:MAX_L+1] >= 2
        # Find first False
        valid = cov[1:MAX_L+1] >= 2
        if not np.any(~valid):
            return MAX_L
        # Find first False
        # np.where(~valid)[0][0] gives index in valid, which is 0-based for 1..MAX_L
        # So L = index + 1
        first_false = np.where(~valid)[0]
        if len(first_false) == 0:
            return MAX_L
        return first_false[0] + 1

    def get_score_from_res(residues):
        cov = compute_cov(residues)
        return get_score(cov)

    # Initial residues: Random or Seed?
    # Seed strategy from previous: ((i+2)//2) % p
    # Let's use random for SA restarts, but maybe start with a decent greedy?
    # Greedy: for each x, pick a prime to cover it?
    # Actually, random init + SA is robust.
    
    best_score_global = 0
    best_res_global = None
    
    num_restarts = 10
    time_per_restart = (deadline - start_time - 5) / num_restarts # 5s safety
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize residues
        # Try a mix of random and "structured"
        if restart == 0:
            residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
        else:
            residues = [random.randint(0, p - 1) for p in primes]
        
        residues = np.array(residues, dtype=np.int64)
        
        cov = compute_cov(residues)
        current_score = get_score(cov)
        
        best_score_local = current_score
        best_res_local = residues.copy()
        
        # Simulated Annealing
        # Temperature schedule: T = T0 * alpha^steps
        # Or time-based? Steps-based is easier.
        # We have time_per_restart seconds.
        # Estimate steps per second.
        # Let's just run until time limit for this restart.
        
        T0 = 1.0
        alpha = 0.999 # Slow cooling
        T = T0
        
        # Precompute weights for prime selection?
        # Smaller primes are more "dense". Changing them has larger impact.
        # Weight by 1/p? Or just uniform?
        # Uniform is simpler.
        
        last_improve_time = time.time()
        
        while time.time() < deadline - 5 - (num_restarts - restart - 1) * time_per_restart:
            if time.time() > deadline - 5:
                break
                
            # Pick a prime index
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            
            # Pick new residue
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Calculate delta in coverage
            # cov_new = cov - cover_arrays[i][old_r] + cover_arrays[i][new_r]
            # We don't want to create a new cov array every time.
            # We can compute the score change directly?
            # Score is determined by the prefix where cov >= 2.
            # Let's just update cov in-place and calculate score.
            
            arr_old = cover_arrays[i][old_r]
            arr_new = cover_arrays[i][new_r]
            
            cov[arr_old] -= 1
            cov[arr_new] += 1
            
            new_score = get_score(cov)
            
            delta = new_score - current_score
            
            if delta > 0:
                # Accept
                current_score = new_score
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_res_local = residues.copy()
                    last_improve_time = time.time()
            else:
                # Metropolis criterion
                # If T is high, accept with prob exp(delta/T)
                # delta is negative
                if T > 0:
                    # delta can be large negative
                    # exp(delta/T) might underflow, which is fine (0)
                    prob = np.exp(delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        # Do NOT update best_res_local unless it's better, 
                        # but we are accepting a worse move, so current_score is worse.
                        # Wait, if we accept a worse move, current_score becomes new_score (worse).
                        # So best_score_local stays the same.
                    else:
                        # Reject
                        cov[arr_new] -= 1
                        cov[arr_old] += 1
                        # current_score remains unchanged
            
            # Cool down
            T *= alpha
            if T < 1e-6:
                T = 1e-6

        # End of SA for this restart
        if best_score_local > best_score_global:
            best_score_global = best_score_local
            best_res_global = best_res_local.copy()
        
        # Optional: Local search polish on best_res_global?
        # Maybe do a simple hill climbing on the global best at the end.

    # Final Polish: Hill Climbing on best_res_global
    if best_res_global is not None and time.time() < deadline - 10:
        residues = best_res_global.copy()
        cov = compute_cov(residues)
        current_score = get_score(cov)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try all primes, best move
            best_delta = 0
            best_i = -1
            best_r = -1
            
            for i in range(n_primes):
                p = primes[i]
                old_r = residues[i]
                # Try a few random residues? Or all?
                # All is too slow (sum p_i ~ 70k).
                # Try 20 random residues per prime.
                for _ in range(20):
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    
                    arr_old = cover_arrays[i][old_r]
                    arr_new = cover_arrays[i][new_r]
                    
                    cov[arr_old] -= 1
                    cov[arr_new] += 1
                    
                    new_score = get_score(cov)
                    delta = new_score - current_score
                    
                    # Revert
                    cov[arr_new] -= 1
                    cov[arr_old] += 1
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_r = new_r
            
            if best_delta > 0:
                i = best_i
                new_r = best_r
                old_r = residues[i]
                arr_old = cover_arrays[i][old_r]
                arr_new = cover_arrays[i][new_r]
                
                cov[arr_old] -= 1
                cov[arr_new] += 1
                residues[i] = new_r
                current_score += best_delta
                improved = True
                if current_score > best_score_global:
                    best_score_global = current_score
                    best_res_global = residues.copy()
            
            # If no improvement, break? Or shuffle?
            # Simple hill climbing stops at local max.
            # To escape, we'd need SA again, but we ran out of time or structure.
            # Let's just stop.

    return [int(r) for r in best_res_global]
