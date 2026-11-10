"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _build_counts_np(residues, primes, max_L):
    """Build count array using numpy for speed."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        counts[start::p] += 1
    return counts

def _find_gap_np(counts, max_L):
    """Find the first gap using numpy."""
    if max_L < 1:
        return 1
    bad = counts[1:max_L+1] < 2
    if not np.any(bad):
        return max_L + 1
    return np.argmax(bad) + 1

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Working range for optimization
    WORK_L = 10000
    
    # Phase 1: Initialization
    # Random initialization
    residues = [random.randint(0, p - 1) for p in primes]
    counts = _build_counts_np(residues, primes, WORK_L)
    
    current_gap = _find_gap_np(counts, WORK_L)
    best_score = current_gap - 1
    best_residues = residues[:]
    
    # Phase 2: Steepest Descent Local Search (0-cost moves)
    # We try to fix the current gap by changing one residue to cover it,
    # without breaking the double coverage of the prefix [1, gap-1].
    
    while time.time() < deadline - 15.0:
        if current_gap > WORK_L:
            break
            
        gap = current_gap
        
        # Identify primes covering gap
        coverers = []
        for i, (p, r) in enumerate(zip(primes, residues)):
            if gap % p == r:
                coverers.append(i)
        
        # We need to find a prime to change to cover gap.
        # Candidates: all primes NOT in coverers.
        best_i = -1
        best_cost = float('inf')
        best_target_r = 0
        best_old_r = 0
        
        for i in range(n_primes):
            if i in coverers:
                continue
            
            p = primes[i]
            old_r = residues[i]
            target_r = gap % p
            
            if old_r == target_r:
                continue # Should not happen if coverers is correct
            
            # Calculate cost: number of x in [1, gap-1] covered by old_r with count == 2
            if old_r == 0:
                first_x = p
            else:
                first_x = old_r
            
            if first_x >= gap:
                cost = 0
            else:
                cost = np.sum(counts[first_x:gap:p] == 2)
            
            if cost < best_cost:
                best_cost = cost
                best_i = i
                best_target_r = target_r
                best_old_r = old_r
                
                if cost == 0:
                    break
        
        if best_i == -1:
            break
        
        # Apply the best move (only if cost is 0 to maintain prefix integrity)
        if best_cost > 0:
            # If no 0-cost move, break to SA phase
            break
        
        p = primes[best_i]
        old_r = residues[best_i]
        new_r = best_target_r
        
        # Update counts
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        
        if start_old <= WORK_L:
            counts[start_old::p] -= 1
            
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
            
        if start_new <= WORK_L:
            counts[start_new::p] += 1
            
        residues[best_i] = new_r
        
        # Update gap
        # Since cost was 0, the prefix [1, gap-1] is intact.
        # The gap must be >= gap + 1.
        current_gap = _find_gap_np(counts, gap)
        
        if current_gap - 1 > best_score:
            best_score = current_gap - 1
            best_residues = residues[:]
            
    # Phase 3: Simulated Annealing to escape local minima
    # Re-initialize from best found so far
    residues = best_residues
    counts = _build_counts_np(residues, primes, WORK_L)
    current_gap = _find_gap_np(counts, WORK_L)
    current_score = current_gap - 1
    
    T = 10.0
    while time.time() < deadline - 2.0:
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Apply move
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old <= WORK_L:
            counts[start_old::p] -= 1
            
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new <= WORK_L:
            counts[start_new::p] += 1
            
        residues[idx] = new_r
        
        new_gap = _find_gap_np(counts, WORK_L)
        new_score = new_gap - 1
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            current_gap = new_gap
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
        else:
            # Revert
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            if start_old <= WORK_L:
                counts[start_old::p] += 1
                
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            if start_new <= WORK_L:
                counts[start_new::p] -= 1
                
            residues[idx] = old_r
            
        T *= 0.999
        if T < 0.01:
            T = 0.01

    residues = best_residues
    residues = [int(r) % p for r, p in zip(residues, primes)]
    
    return residues
# EVOLVE-BLOCK-END
