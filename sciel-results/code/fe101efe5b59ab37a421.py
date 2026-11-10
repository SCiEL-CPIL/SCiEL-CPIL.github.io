"""
Strategy: Vectorized Simulated Annealing with Bottleneck-Driven Mutation
The current program uses a sequential backtracking solver. This new approach uses
NumPy to vectorize the coverage calculation and a simulated annealing loop.
Key differences:
1. Vectorized Evaluation: We maintain a coverage array `cov` of length L_max.
   Changing one residue `r_i` from `old` to `new` involves subtracting a pattern
   and adding a pattern. We can compute the change in score efficiently.
2. Bottleneck-Driven Mutation: Instead of random flips, we identify the first
   integer `x_fail` where `cov[x_fail] < 2`. We prioritize mutating primes that
   cover `x_fail` (i.e., primes `p` where `x_fail % p` is a candidate residue).
3. Annealing: We accept worse moves early to escape local minima, then cool down.
"""

import time
import numpy as np
import random

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
    # Safety margin: return before 140s to ensure <150s wall clock
    deadline = start_time + 135.0
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    primes_np = np.array(primes, dtype=np.int64)
    
    # We want to maximize L. Let's search up to L_max.
    # If we can easily cover 1000, we might push further.
    # Start with a conservative L_max that allows fast vectorization.
    # 10,000 is a good balance. If we reach it, we can try to extend.
    L_max = 10000
    
    # Initialize residues
    # Seed strategy: r_p = (i+2)//2 % p
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    def compute_coverage(res):
        """Compute coverage array for [1, L_max] given residues."""
        cov = np.zeros(L_max + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_np[i]
            r = res[i]
            # x = r + k*p
            # If r == 0, first x is p. Else first x is r.
            if r == 0:
                start = p
            else:
                start = r
            if start > L_max:
                continue
            # Vectorized addition
            # Indices: start, start+p, start+2p, ... <= L_max
            indices = np.arange(start, L_max + 1, p)
            cov[indices] += 1
        return cov

    def get_score(cov):
        """Find largest L such that cov[1..L] >= 2."""
        # Find first index where cov < 2
        # cov[0] is unused.
        # We want the first i in 1..L_max where cov[i] < 2.
        # If none, return L_max.
        # Note: cov is int32.
        # Create a boolean mask where cov < 2
        # We need to check from 1 onwards.
        # np.argmax returns 0 if no element is True.
        # So we check if any element in cov[1:] < 2.
        if not np.any(cov[1:] < 2):
            return L_max
        # Find the first index in cov[1:] where condition is true
        # argmax on (cov[1:] < 2) gives index relative to cov[1:]
        # Add 1 to get actual index
        first_fail = np.argmax(cov[1:] < 2) + 1
        return first_fail - 1

    # Initial state
    cov = compute_coverage(residues)
    best_score = get_score(cov)
    best_residues = residues.copy()
    
    # Precompute indices for each prime to speed up updates
    # For prime i with residue r, the indices are start, start+p, ...
    # We can store the "pattern" of indices for each possible residue? No, too big.
    # Instead, we calculate indices on the fly but use numpy arange which is fast.
    
    # Annealing parameters
    T = 50.0  # Initial temperature
    cooling = 0.9995
    min_T = 0.5
    
    # Helper to get indices for a prime/residue
    def get_indices(p, r):
        if r == 0:
            start = p
        else:
            start = r
        if start > L_max:
            return np.array([], dtype=np.int64)
        return np.arange(start, L_max + 1, p)

    # Main Loop
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Determine mutation strategy
        # Find current bottleneck
        # If best_score < L_max, we know where it fails.
        # Let's find the failure point in the current cov
        if np.any(cov[1:] < 2):
            fail_idx = np.argmax(cov[1:] < 2) + 1
        else:
            fail_idx = L_max + 1
            
        # Choose a prime to mutate
        # Strategy:
        # 1. If fail_idx <= L_max, prefer primes that cover fail_idx.
        #    A prime p covers fail_idx if (fail_idx % p) is a valid residue.
        #    Actually, we can set residue to fail_idx % p to cover it.
        # 2. Otherwise, pick a random prime.
        
        if fail_idx <= L_max:
            # Find primes that can cover fail_idx
            # All primes can cover it by setting r = fail_idx % p.
            # But we want to prioritize primes where this change is "cheap" or "helpful".
            # Let's pick a random prime from the first 50 primes (small p = more frequent)
            # or specifically those that currently do NOT cover fail_idx but could.
            
            # Let's pick a random prime index
            i = random.randint(0, n_primes - 1)
            p = primes_np[i]
            
            # If this prime already covers fail_idx, maybe pick another?
            # If cov[fail_idx] < 2, we need MORE coverage.
            # If p already covers fail_idx, changing its residue might remove coverage.
            # So we should prefer primes that do NOT currently cover fail_idx,
            # and set their residue to fail_idx % p.
            
            current_r = residues[i]
            if current_r == 0:
                covers_fail = (fail_idx % p == 0 and fail_idx >= p)
            else:
                covers_fail = (fail_idx % p == current_r and fail_idx >= current_r)
                
            if covers_fail:
                # This prime already helps. Changing it might hurt.
                # Let's try a different prime that doesn't cover it.
                # Retry a few times
                for _ in range(5):
                    i = random.randint(0, n_primes - 1)
                    p = primes_np[i]
                    current_r = residues[i]
                    if current_r == 0:
                        covers_fail = (fail_idx % p == 0 and fail_idx >= p)
                    else:
                        covers_fail = (fail_idx % p == current_r and fail_idx >= current_r)
                    if not covers_fail:
                        break
                # If we still have a covering prime, just use it but be careful
                # Or pick a random non-covering prime if possible
                # For simplicity, proceed with i, but maybe choose new_r carefully
            else:
                # This prime does not cover fail_idx.
                # Set new_r to cover it.
                new_r = fail_idx % p
        else:
            # No bottleneck in current range, or we are exploring.
            i = random.randint(0, n_primes - 1)
            p = primes_np[i]
            new_r = random.randint(0, p - 1)
            
        old_r = residues[i]
        
        # If new_r is same as old_r, skip
        if new_r == old_r:
            continue
            
        # Calculate new coverage
        # cov_new = cov - old_pattern + new_pattern
        
        old_indices = get_indices(p, old_r)
        new_indices = get_indices(p, new_r)
        
        # Apply change
        if len(old_indices) > 0:
            cov[old_indices] -= 1
        if len(new_indices) > 0:
            cov[new_indices] += 1
        
        # Calculate new score
        new_score = get_score(cov)
        
        # Acceptance
        if new_score >= best_score:
            # Accept
            residues[i] = new_r
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
        else:
            # Metropolis criterion
            # Delta score is negative.
            # Probability = exp(delta / T)
            # Since score is integer, delta is at most -1, -2...
            # For large T, we accept more.
            if T > 0:
                delta = new_score - best_score
                # If delta is very negative, exp will be 0
                if delta > -100: 
                    prob = np.exp(delta / T)
                    if random.random() < prob:
                        residues[i] = new_r
                    else:
                        # Reject: revert
                        if len(old_indices) > 0:
                            cov[old_indices] += 1
                        if len(new_indices) > 0:
                            cov[new_indices] -= 1
                else:
                    # Reject
                    if len(old_indices) > 0:
                        cov[old_indices] += 1
                    if len(new_indices) > 0:
                        cov[new_indices] -= 1
            else:
                # Reject
                if len(old_indices) > 0:
                    cov[old_indices] += 1
                if len(new_indices) > 0:
                    cov[new_indices] -= 1
        
        # Cool down
        if T > min_T:
            T *= cooling
            
        # If we reached L_max, try to extend?
        # For now, just keep optimizing within L_max.
        # If best_score == L_max, we can try to increase L_max.
        if best_score == L_max and L_max < 20000:
            # Extend L_max
            old_L_max = L_max
            L_max = min(L_max * 2, 50000)
            # Recompute coverage for the new range
            # We need to append to cov
            new_cov = np.zeros(L_max + 1, dtype=np.int32)
            # Copy old
            new_cov[:old_L_max+1] = cov
            # Add contributions for the new range
            for i in range(n_primes):
                p = primes_np[i]
                r = residues[i]
                # We need to add indices from old_L_max+1 to L_max
                # Find first index >= old_L_max+1
                # x = r + k*p
                # k_min = ceil((old_L_max+1 - r) / p)
                # If r == 0, x is multiple of p.
                
                # Simpler: just recompute full coverage for the new L_max
                # This is O(N * L_max/p). For L_max=20000, it's fast.
                cov = compute_coverage(residues)
                break # Just recompute once
            # Update best score check
            # best_score might be higher now
            # Actually, get_score uses L_max global variable, so it will check further.
            # But best_score was capped at old L_max.
            # Let's re-evaluate best_score
            current_score = get_score(cov)
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()

    return best_residues.tolist()

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    # Quick score check
    primes = _primes_upto(1000)
    cov = np.zeros(10001, dtype=np.int32)
    for i, p in enumerate(primes):
        r = res[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= 10000:
            indices = np.arange(start, 10001, p)
            cov[indices] += 1
    L = 0
    for x in range(1, 10001):
        if cov[x] >= 2:
            L = x
        else:
            break
    print(f"Score: {L}")
