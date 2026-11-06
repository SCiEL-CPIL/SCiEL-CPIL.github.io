"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _build_coverage(primes, residues, W):
    """Build coverage array for x in [0, W]."""
    coverage = np.zeros(W + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= W:
            coverage[start::p] += 1
    return coverage

def _find_L(coverage, W):
    """Find largest L such that coverage[x] >= 2 for all x in 1..L."""
    # Check from 1 to W
    for x in range(1, W + 1):
        if coverage[x] < 2:
            return x - 1
    return W

def _greedy_construct(primes, W=5000):
    """Greedy forward pass."""
    n_primes = len(primes)
    residues = np.zeros(n_primes, dtype=np.int64)
    coverage = np.zeros(W + 1, dtype=np.int32)
    
    for idx, p in enumerate(primes):
        best_r = 0
        best_score = -1
        mask_one = (coverage == 1)
        
        for r in range(p):
            start = r if r > 0 else p
            if start > W:
                score = 0
            else:
                score = np.sum(mask_one[start::p])
            
            if score > best_score:
                best_score = score
                best_r = r
            elif score == best_score and r < best_r:
                best_r = r
        
        residues[idx] = best_r
        start = best_r if best_r > 0 else p
        if start <= W:
            coverage[start::p] += 1
    
    return residues

def _local_search(primes, residues, W=5000, time_limit=None, start_time=None, aggressive=True):
    """Aggressive local search targeting the first gap."""
    n_primes = len(primes)
    coverage = _build_coverage(primes, residues, W)
    L = _find_L(coverage, W)
    current_residues = residues.copy()
    current_coverage = coverage.copy()
    current_L = L
    
    if start_time is None:
        start_time = time.time()
    
    gap_idx = current_L + 1
    
    max_iter = 200000
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        if gap_idx > W:
            break
        
        # Try to fix the gap by changing one prime's residue
        # Choose primes that could cover gap_idx
        # A prime p covers gap_idx if we set r_p = gap_idx % p
        # We need to check if this change breaks the prefix [1, gap_idx-1]
        
        # To be efficient, try a few random primes
        improved = False
        for _ in range(10):
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = current_residues[i]
            target_r = gap_idx % p
            
            if target_r == old_r:
                continue
            
            # Check validity: no x in [1, gap_idx-1] drops below 2
            # Losing positions: old_r, old_r+p, ... < gap_idx
            if old_r > 0:
                pos_lose = np.arange(old_r, gap_idx, p)
            else:
                pos_lose = np.arange(p, gap_idx, p)
            
            if len(pos_lose) > 0 and np.any(current_coverage[pos_lose] == 2):
                continue
            
            # Apply change
            if old_r > 0:
                current_coverage[old_r::p] -= 1
            else:
                current_coverage[p::p] -= 1
            
            if target_r > 0:
                current_coverage[target_r::p] += 1
            else:
                current_coverage[p::p] += 1
            
            current_residues[i] = target_r
            
            # Check new L
            # Only need to check from 1 to min(gap_idx+10, W) to see if it extends
            # But to be safe, find the new gap
            new_L = current_L
            for x in range(1, min(gap_idx + 50, W + 1)):
                if current_coverage[x] < 2:
                    new_L = x - 1
                    break
            else:
                # No gap found up to min(gap_idx+50, W)
                # Check further
                for x in range(gap_idx + 50, W + 1):
                    if current_coverage[x] < 2:
                        new_L = x - 1
                        break
                else:
                    new_L = W
            
            if new_L > current_L:
                current_L = new_L
                gap_idx = current_L + 1
                improved = True
                break
            else:
                # Revert
                if target_r > 0:
                    current_coverage[target_r::p] -= 1
                else:
                    current_coverage[p::p] -= 1
                
                if old_r > 0:
                    current_coverage[old_r::p] += 1
                else:
                    current_coverage[p::p] += 1
                
                current_residues[i] = old_r
        
        if not improved and aggressive:
            # Allow neutral moves to escape local minima
            # Try a random change that doesn't break the prefix
            for _ in range(5):
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                old_r = current_residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                # Check if prefix [1, current_L] remains valid
                if old_r > 0:
                    pos_lose = np.arange(old_r, current_L + 1, p)
                else:
                    pos_lose = np.arange(p, current_L + 1, p)
                
                if len(pos_lose) > 0 and np.any(current_coverage[pos_lose] == 2):
                    continue
                
                # Apply
                if old_r > 0:
                    current_coverage[old_r::p] -= 1
                else:
                    current_coverage[p::p] -= 1
                
                if new_r > 0:
                    current_coverage[new_r::p] += 1
                else:
                    current_coverage[p::p] += 1
                
                current_residues[i] = new_r
                
                # Verify prefix is still valid
                valid = True
                for x in range(1, current_L + 1):
                    if current_coverage[x] < 2:
                        valid = False
                        break
                
                if valid:
                    break
                else:
                    # Revert
                    if new_r > 0:
                        current_coverage[new_r::p] -= 1
                    else:
                        current_coverage[p::p] -= 1
                    
                    if old_r > 0:
                        current_coverage[old_r::p] += 1
                    else:
                        current_coverage[p::p] += 1
                    
                    current_residues[i] = old_r
    
    return current_residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    W = 5000
    
    # Phase 1: Greedy construction
    residues = _greedy_construct(primes, W)
    coverage = _build_coverage(primes, residues, W)
    L = _find_L(coverage, W)
    
    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        residues, L = _local_search(primes, residues, W, 
                                     time_limit=deadline - time.time() - 5.0,
                                     start_time=start_time,
                                     aggressive=True)
    
    # Phase 3: Multiple restarts if time allows
    best_residues = residues.copy()
    best_L = L
    
    while time.time() < deadline - 15.0:
        # Restart from perturbed version of best solution
        perturbed = best_residues.copy()
        # Perturb a few random residues
        for _ in range(3):
            i = random.randint(0, n_primes - 1)
            perturbed[i] = random.randint(0, primes[i] - 1)
        
        # Run local search from perturbed start
        res, L_new = _local_search(primes, perturbed, W,
                                    time_limit=deadline - time.time() - 10.0,
                                    start_time=start_time,
                                    aggressive=True)
        
        if L_new > best_L:
            best_L = L_new
            best_residues = res
    
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
