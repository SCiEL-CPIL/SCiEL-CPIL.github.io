import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate primes <= 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Working window - we'll extend as needed
    W = 50000
    
    best_L_found = 0
    best_residues = None
    
    def compute_L(cov):
        """Find the largest L such that cov[1..L] >= 2"""
        mask = cov[1:] < 2
        if not mask.any():
            return W
        first_bad_idx = np.argmax(mask)
        return first_bad_idx
    
    def compute_coverage(residues):
        """Compute coverage array from residues"""
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r > 0 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov
    
    def try_improve(residues, cov, current_L):
        """
        Try to extend the doubly-covered prefix by fixing gaps.
        Returns (improved, new_residues, new_cov, new_L)
        """
        if current_L >= W:
            return False, residues, cov, current_L
        
        bad_pos = current_L + 1
        # We need at least 2 primes to cover bad_pos
        # Currently, how many primes cover bad_pos?
        current_coverage_at_bad = cov[bad_pos]
        
        if current_coverage_at_bad >= 2:
            # This shouldn't happen if compute_L is correct, but just in case
            return False, residues, cov, current_L
        
        # We need to add coverage at bad_pos.
        # Strategy: find a prime that currently covers bad_pos (if any) and 
        # try to find another prime that can be adjusted to also cover bad_pos
        # without breaking the prefix.
        
        # First, identify which primes already cover bad_pos
        covering_primes = []
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            if bad_pos % p == r:
                covering_primes.append(i)
        
        # If no prime covers bad_pos, we need to change at least one.
        # If one prime covers bad_pos, we need to change another.
        
        # Try to find a prime i (not in covering_primes) such that:
        # 1. Setting residues[i] = bad_pos % p_i doesn't break the prefix [1, bad_pos-1]
        # 2. This is checked by ensuring that for all j in [1, bad_pos-1],
        #    if j was covered only by prime i (cov[j] == 1 and prime i covers j),
        #    then we'd break it. But actually, we need cov[j] >= 2 for all j < bad_pos.
        #    So if we remove prime i's coverage at some j < bad_pos where cov[j] == 2,
        #    it becomes 1, which is bad.
        
        # For each candidate prime i not in covering_primes:
        best_i = -1
        best_new_r = -1
        best_score = float('inf')
        
        for i in range(n_primes):
            if i in covering_primes:
                continue
            
            p = primes_arr[i]
            old_r = residues[i]
            new_r = bad_pos % p
            
            if new_r == old_r:
                # This prime already covers bad_pos? But we said it's not in covering_primes.
                # This can happen if bad_pos % p == old_r, which means it does cover bad_pos.
                # So this case shouldn't occur. Skip.
                continue
            
            # Check if changing residue i from old_r to new_r breaks the prefix
            # Damage: positions j in [1, bad_pos-1] where:
            #   - prime i covers j with old_r (i.e., j % p == old_r)
            #   - cov[j] == 2 (so removing prime i makes it 1)
            
            # Efficient check: iterate through positions covered by old_r in [1, bad_pos-1]
            old_start = old_r if old_r > 0 else p
            if old_start > bad_pos - 1:
                # No positions in [1, bad_pos-1] covered by old_r
                damage = 0
            else:
                # Count positions j in [old_start, bad_pos-1] step p where cov[j] == 2
                damage = 0
                j = old_start
                while j < bad_pos:
                    if cov[j] == 2:
                        damage += 1
                    j += p
            
            # Also check gain: positions in [1, bad_pos-1] that gain coverage
            # (not critical for prefix, but good to know)
            
            # Score: minimize damage
            if damage < best_score:
                best_score = damage
                best_i = i
                best_new_r = new_r
        
        if best_i == -1:
            # No good candidate found. Try a random change to escape local optimum.
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            new_r = bad_pos % p
            if new_r == residues[i]:
                return False, residues, cov, current_L
            best_i = i
            best_new_r = new_r
        
        # Apply the change
        i = best_i
        p = primes_arr[i]
        old_r = residues[i]
        new_r = best_new_r
        
        old_start = old_r if old_r > 0 else p
        new_start = new_r if new_r > 0 else p
        
        new_cov = cov.copy()
        if old_start <= W:
            new_cov[old_start:W+1:p] -= 1
        if new_start <= W:
            new_cov[new_start:W+1:p] += 1
        
        new_residues = residues.copy()
        new_residues[i] = new_r
        
        new_L = compute_L(new_cov)
        
        if new_L > current_L:
            return True, new_residues, new_cov, new_L
        else:
            return False, residues, cov, current_L
    
    def run_restart():
        # Smart initialization: start with residues that cover small numbers well
        # For each prime p, choose r_p such that it covers a small number.
        # Simple approach: r_p = 1 for all p (covers 1, 1+p, 1+2p, ...)
        # Or: r_p = p - 1 (covers p-1, 2p-1, ...)
        
        # Let's try a few different initializations
        init_type = random.randint(0, 2)
        if init_type == 0:
            residues = np.array([1] * n_primes, dtype=np.int64)
        elif init_type == 1:
            residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int64)
        else:
            # Cover small numbers: for prime i, set r_p = (i+1) % p
            residues = np.array([(i+1) % p for i, p in enumerate(primes)], dtype=np.int64)
        
        cov = compute_coverage(residues)
        current_L = compute_L(cov)
        local_best_L = current_L
        local_best_res = residues.copy()
        
        # Iterative improvement
        max_steps = 1000
        no_improve_count = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if current_L >= W:
                break
            
            improved, residues, cov, current_L = try_improve(residues, cov, current_L)
            
            if improved:
                no_improve_count = 0
                if current_L > local_best_L:
                    local_best_L = current_L
                    local_best_res = residues.copy()
            else:
                no_improve_count += 1
                # If stuck for a while, try a random perturbation
                if no_improve_count > 50:
                    # Perturb a random residue
                    i = random.randint(0, n_primes - 1)
                    p = primes_arr[i]
                    old_r = residues[i]
                    new_r = random.randint(0, p-1)
                    if new_r != old_r:
                        old_start = old_r if old_r > 0 else p
                        new_start = new_r if new_r > 0 else p
                        if old_start <= W:
                            cov[old_start:W+1:p] -= 1
                        if new_start <= W:
                            cov[new_start:W+1:p] += 1
                        residues[i] = new_r
                        current_L = compute_L(cov)
                        no_improve_count = 0
        
        return local_best_L, local_best_res
    
    # Main Loop: Multiple Restarts
    restarts = 0
    while time.time() < deadline - 10:
        L, res = run_restart()
        if L > best_L_found:
            best_L_found = L
            best_residues = res
            # print(f"New best: {L}, restart {restarts}", file=sys.stderr)
        
        restarts += 1
        
        # If we've made good progress, try to extend with more focused search
        if best_L_found > 100 and restarts % 5 == 0:
            # Do a focused search from the best solution
            residues = best_residues.copy()
            cov = compute_coverage(residues)
            current_L = compute_L(cov)
            
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                if current_L >= W:
                    break
                improved, residues, cov, current_L = try_improve(residues, cov, current_L)
                if improved and current_L > best_L_found:
                    best_L_found = current_L
                    best_residues = residues.copy()
    
    if best_residues is None:
        # Fallback: all 1s
        return [1] * n_primes
        
    return [int(r) for r in best_residues]
