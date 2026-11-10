"""
Optimized doubly-covered prefix via vectorized Simulated Annealing.
Uses NumPy for fast count updates and score evaluation.
Includes pairwise moves to escape local minima.
"""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)
_L_MAX = 10000  # We search for L up to 10000, though target is 1000+
# We need to be careful with memory and speed. 10000 is reasonable for numpy.

def _build_count_np(residues, L_MAX):
    """Build count array for x in [1, L_MAX] using NumPy."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # We want the first index where count < 2.
    # count[0] is ignored.
    bad_indices = np.where(count[1:L_MAX+1] < 2)[0]
    if len(bad_indices) == 0:
        return L_MAX
    return bad_indices[0]

def _score_np(residues, L_MAX):
    """Compute the score (largest L with full double coverage)."""
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction: assign residues to maximize coverage in the early range.
    We focus on the first 2000 integers to establish a strong base.
    """
    residues = [0] * _NPRIMES
    L_LOCAL = 2000
    count = np.zeros(L_LOCAL + 1, dtype=np.int32)
    
    # Process primes in increasing order
    for idx in range(_NPRIMES):
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1e9
        
        # For small primes, try all residues. For large, sample.
        if p <= 100:
            candidate_rs = range(p)
        else:
            # Sample residues, ensuring we try 0 and 1
            candidate_rs = set([random.randint(0, p - 1) for _ in range(20)])
            candidate_rs.add(0)
            candidate_rs.add(1)
            candidate_rs = list(candidate_rs)
        
        for r in candidate_rs:
            start = p if r == 0 else r
            if start > L_LOCAL:
                # If start is beyond our local window, it contributes 0 to this window
                gain = 0
            else:
                # Calculate gain: prioritize filling holes (count 0 or 1)
                # We want to maximize the number of x with count[x] >= 2
                # Or more simply, minimize the number of x with count[x] < 2
                # Let's just sum up the "value" of covering each point
                # Value is high if count is low.
                segment = count[start::p]
                # If segment is empty, gain is 0
                if len(segment) == 0:
                    gain = 0
                else:
                    # Heuristic: +10 for count 0, +5 for count 1, +1 for count 2, -1 for count >= 3
                    # This encourages balancing coverage.
                    gain = np.sum(np.where(segment == 0, 10, 
                           np.where(segment == 1, 5, 
                           np.where(segment == 2, 1, -1))))
            
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_LOCAL:
            count[start::p] += 1
    
    # Extend to full L_MAX to get a true baseline
    full_count = _build_count_np(residues, _L_MAX)
    current_L = _find_prefix_L_np(full_count, _L_MAX)
    
    return residues, current_L

def _simulated_annealing(residues, deadline, L_MAX=5000, T0=50.0, cooling=0.999, max_steps=20000):
    """
    Simulated Annealing with NumPy optimizations.
    Moves:
    1. Single residue change (biased to small primes).
    2. Pairwise change (if single moves stagnate).
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    best_L = current_L
    best_residues = list(residues)
    
    T = T0
    step = 0
    stall_count = 0
    
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # Choose move type
        # 80% single, 20% double (if stalled)
        if stall_count > 50 and random.random() < 0.3:
            move_type = 'double'
        else:
            move_type = 'single'
            
        if move_type == 'single':
            # Bias towards small primes
            if random.random() < 0.8:
                idx = random.randint(0, min(50, _NPRIMES - 1))
            else:
                idx = random.randint(0, _NPRIMES - 1)
                
            p = _PRIMES[idx]
            old_r = residues[idx]
            
            # Choose new residue
            if random.random() < 0.4:
                # Targeted: try to cover the current gap
                gap = current_L + 1
                if gap <= L_MAX:
                    new_r = gap % p
                else:
                    new_r = random.randint(0, p - 1)
            else:
                new_r = random.randint(0, p - 1)
                
            if new_r == old_r:
                continue
            
            # Apply move
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            if old_start <= L_MAX:
                count[old_start::p] -= 1
            if new_start <= L_MAX:
                count[new_start::p] += 1
                
            new_L = _find_prefix_L_np(count, L_MAX)
            
            # Accept/Reject
            if new_L >= current_L:
                residues[idx] = new_r
                current_L = new_L
                stall_count = 0
                if new_L > best_L:
                    best_L = new_L
                    best_residues = list(residues)
            else:
                delta = current_L - new_L
                prob = math.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    residues[idx] = new_r
                    current_L = new_L
                    stall_count += 1
                else:
                    # Undo
                    if new_start <= L_MAX:
                        count[new_start::p] -= 1
                    if old_start <= L_MAX:
                        count[old_start::p] += 1
                    stall_count += 1
        else:
            # Double move: change two distinct primes
            idx1, idx2 = random.sample(range(_NPRIMES), 2)
            # Bias towards small primes for at least one
            if idx1 > 50 and idx2 > 50:
                if random.random() < 0.5:
                    idx1 = random.randint(0, 50)
                else:
                    idx2 = random.randint(0, 50)
                    
            p1 = _PRIMES[idx1]
            p2 = _PRIMES[idx2]
            old_r1 = residues[idx1]
            old_r2 = residues[idx2]
            
            new_r1 = random.randint(0, p1 - 1)
            new_r2 = random.randint(0, p2 - 1)
            
            if new_r1 == old_r1 and new_r2 == old_r2:
                continue
            
            # Apply
            s1_old = p1 if old_r1 == 0 else old_r1
            s1_new = p1 if new_r1 == 0 else new_r1
            s2_old = p2 if old_r2 == 0 else old_r2
            s2_new = p2 if new_r2 == 0 else new_r2
            
            if s1_old <= L_MAX: count[s1_old::p1] -= 1
            if s1_new <= L_MAX: count[s1_new::p1] += 1
            if s2_old <= L_MAX: count[s2_old::p2] -= 1
            if s2_new <= L_MAX: count[s2_new::p2] += 1
            
            new_L = _find_prefix_L_np(count, L_MAX)
            
            if new_L >= current_L:
                residues[idx1] = new_r1
                residues[idx2] = new_r2
                current_L = new_L
                stall_count = 0
                if new_L > best_L:
                    best_L = new_L
                    best_residues = list(residues)
            else:
                delta = current_L - new_L
                prob = math.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    residues[idx1] = new_r1
                    residues[idx2] = new_r2
                    current_L = new_L
                    stall_count += 1
                else:
                    # Undo
                    if s1_new <= L_MAX: count[s1_new::p1] -= 1
                    if s1_old <= L_MAX: count[s1_old::p1] += 1
                    if s2_new <= L_MAX: count[s2_new::p2] -= 1
                    if s2_old <= L_MAX: count[s2_old::p2] += 1
                    stall_count += 1
        
        T *= cooling
        
        # Periodically check if we can extend L_MAX? No, keep fixed for speed.
        # But if we reach L_MAX, we might want to stop or expand.
        # For now, if L == L_MAX, we can stop early as it's a strong result.
        if current_L == L_MAX:
            break

    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Phase 1: Greedy Initialization
    residues, L = _greedy_construct(deadline - 120)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Simulated Annealing
    # Run multiple rounds with different temps or just one long run?
    # One long run with adaptive cooling is usually better.
    residues, L = _simulated_annealing(residues, deadline - 10, L_MAX=5000, T0=100.0, cooling=0.9995)
    
    if L > best_L:
        best_L = L
        best_residues = list(residues)
        
    # Phase 3: Final Polish with larger L_MAX to verify and potentially find longer prefix
    # If best_L is close to 5000, we might have truncated it.
    # Let's do a final check with L_MAX=10000 on the best found so far.
    if time.time() < deadline - 5:
        final_L = _score_np(best_residues, 10000)
        if final_L > best_L:
            best_L = final_L
            
        # If we found a high L, do a quick local search to extend it?
        # The SA already did this. Let's just return.
        
    return best_residues

# EVOLVE-BLOCK-END
