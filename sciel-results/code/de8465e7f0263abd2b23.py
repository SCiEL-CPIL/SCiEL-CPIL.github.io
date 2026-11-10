import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s kill

    # Generate primes up to 1000
    def get_primes(n):
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]
    
    primes = get_primes(1000)
    n_primes = len(primes)  # 168
    p_list = primes
    n_primes = len(p_list)

    # Precompute index arrays for each prime for faster coverage updates
    # idx_arrays[i] = np.array of indices in [1, limit] that are ≡ 0 (mod p) offset by residue
    # Actually, for a given prime p and residue r, the covered positions in 1..limit are:
    # If r == 0: p, 2p, 3p, ...
    # If r > 0: r, r+p, r+2p, ...
    # We'll use a different approach: precompute for each prime the list of base offsets

    def compute_coverage(residues, limit):
        """Compute coverage for x in 1..limit."""
        cov = np.zeros(limit + 1, dtype=np.int16)
        for i, p in enumerate(p_list):
            r = int(residues[i])
            if r == 0:
                start = p
            else:
                start = r
            if start <= limit:
                cov[start::p] += 1
        return cov

    def score_from_coverage(cov, limit):
        """Find largest L such that cov[x] >= 2 for all x in 1..L."""
        bad_indices = np.where(cov[1:limit+1] < 2)[0]
        if len(bad_indices) == 0:
            return limit
        return int(bad_indices[0])

    def full_score(residues, limit=100000):
        cov = compute_coverage(residues, limit)
        return score_from_coverage(cov, limit)

    # Phase 1: Greedy forward pass
    # For each prime in order, pick residue that maximizes the length of the doubly-covered prefix
    # We work on a window of size W during greedy, then refine
    W_GREEDY = 30000

    def greedy_construct():
        residues = np.zeros(n_primes, dtype=np.int32)
        cov = np.zeros(W_GREEDY + 1, dtype=np.int16)
        
        for i, p in enumerate(p_list):
            if time.time() > deadline - 30:
                break
            best_r = 0
            best_score = -1
            # Try all residues for this prime
            for r in range(p):
                # Apply residue r
                if r == 0:
                    start = p
                else:
                    start = r
                if start <= W_GREEDY:
                    cov[start::p] += 1
                
                # Compute score on current coverage
                # Only need to check up to W_GREEDY
                bad = np.where(cov[1:W_GREEDY+1] < 2)[0]
                if len(bad) == 0:
                    s = W_GREEDY
                else:
                    s = int(bad[0])
                
                if s > best_score:
                    best_score = s
                    best_r = r
                elif s == best_score:
                    # Tie-break: prefer smaller residue? Or random?
                    pass
                
                # Revert
                if start <= W_GREEDY:
                    cov[start::p] -= 1
            
            residues[i] = best_r
            # Commit
            p_val = p_list[i]
            r_val = int(best_r)
            if r_val == 0:
                st = p_val
            else:
                st = r_val
            if st <= W_GREEDY:
                cov[st::p_val] += 1
        
        return residues

    # Phase 2: Local search with multiple restarts
    # Use a smaller window for speed during local search, then full eval at the end

    def local_search(residues, max_time=20.0):
        current = residues.copy()
        limit = 50000
        cov = compute_coverage(current, limit)
        current_score = score_from_coverage(cov, limit)
        
        search_start = time.time()
        improved = True
        stall_count = 0
        max_stall = 80
        
        while improved and (time.time() - search_start) < max_time and time.time() < deadline - 15:
            improved = False
            stall_count = 0
            
            # Shuffle prime order for diversity
            order = list(range(n_primes))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 15:
                    break
                
                p = p_list[i]
                old_r = int(current[i])
                
                # Try several candidate residues
                candidates = []
                # Try the old residue's neighbors
                candidates.append((old_r - 1) % p)
                candidates.append((old_r + 1) % p)
                # Try a few random
                for _ in range(4):
                    candidates.append(random.randint(0, p - 1))
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    
                    if old_r == 0:
                        old_start = p
                    else:
                        old_start = old_r
                    if new_r == 0:
                        new_start = p
                    else:
                        new_start = new_r
                    
                    # Update coverage
                    if old_start <= limit:
                        cov[old_start::p] -= 1
                    if new_start <= limit:
                        cov[new_start::p] += 1
                    
                    new_score = score_from_coverage(cov, limit)
                    
                    if new_score > current_score:
                        current[i] = new_r
                        current_score = new_score
                        improved = True
                        stall_count = 0
                    else:
                        # Revert
                        if new_start <= limit:
                            cov[new_start::p] -= 1
                        if old_start <= limit:
                            cov[old_start::p] += 1
                        stall_count += 1
                        if stall_count > max_stall:
                            break
                
                if stall_count > max_stall:
                    break
        
        return current, current_score

    # Run greedy construction
    best_residues = greedy_construct()
    best_score = full_score(best_residues, 100000)
    
    # Multiple restarts with local search
    num_restarts = 0
    max_restarts = 6
    
    while time.time() < deadline - 20 and num_restarts < max_restarts:
        num_restarts += 1
        
        if num_restarts == 1:
            start_res = best_residues.copy()
        else:
            # Perturb best solution
            start_res = best_residues.copy()
            n_perturb = random.randint(10, 40)
            for _ in range(n_perturb):
                i = random.randint(0, n_primes - 1)
                start_res[i] = random.randint(0, p_list[i] - 1)
        
        # Time budget per restart
        remaining = deadline - 20 - time.time()
        if remaining < 10:
            break
        
        time_per_restart = min(18.0, remaining * 0.8)
        result_res, result_score = local_search(start_res, max_time=time_per_restart)
        
        # Full evaluation
        final_score = full_score(result_res, 100000)
        
        if final_score > best_score:
            best_score = final_score
            best_residues = result_res.copy()
            print(f"Restart {num_restarts}: improved to {best_score}")
    
    # Final check with best residues
    final_check = full_score(best_residues, 100000)
    if final_check > best_score:
        best_score = final_check
    
    return best_residues.tolist()
