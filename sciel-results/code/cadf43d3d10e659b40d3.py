import time
import random
import numpy as np

BASE = 51975

def _get_divisors():
    n = BASE
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

ALL_MODULI = _get_divisors()
NUM_MODULI = len(ALL_MODULI)

def _calc_score_and_counts(residues):
    counts = np.zeros(BASE, dtype=np.int32)
    for i, m in enumerate(ALL_MODULI):
        r = residues[i]
        counts[r::m] += 1
    score = np.sum(counts == 0)
    return score, counts

def _greedy_init(order_indices, rng):
    residues = [0] * NUM_MODULI
    counts = np.zeros(BASE, dtype=np.int32)
    # Process moduli in the given order
    for idx in order_indices:
        m = ALL_MODULI[idx]
        best_r = 0
        best_gain = -1
        # Try all residues for this modulus
        # To speed up, we can sample if m is large, but m is small enough (max 51975)
        # However, iterating m times for each of 47 moduli is 47 * 51975 ~ 2.4M ops, which is fine.
        # But we only do this once per init.
        for r in range(m):
            # Gain is number of 0s in counts[r::m]
            gain = np.sum(counts[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[idx] = best_r
        counts[best_r::m] += 1
        
    score = np.sum(counts == 0)
    return residues, counts, score

def _simulated_annealing(residues, counts, score, deadline, rng):
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    current_residues = residues[:]
    current_counts = counts.copy()
    current_score = score
    
    best_residues = current_residues[:]
    best_score = current_score
    best_counts = current_counts.copy()
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick random modulus
        i = rng.randint(0, NUM_MODULI - 1)
        m = ALL_MODULI[i]
        old_r = current_residues[i]
        
        # Pick new random residue
        new_r = rng.randint(0, m - 1)
        if new_r == old_r:
            continue
            
        # Calculate delta
        # Loss: residues in old slice that are 1
        # Gain: residues in new slice that are 0
        # Note: old and new slices are disjoint
        loss = np.sum(current_counts[old_r::m] == 1)
        gain = np.sum(current_counts[new_r::m] == 0)
        delta = loss - gain
        
        # Acceptance
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            current_counts[old_r::m] -= 1
            current_counts[new_r::m] += 1
            current_residues[i] = new_r
            current_score += delta
            
            if current_score < best_score:
                best_score = current_score
                best_residues = current_residues[:]
                best_counts = current_counts.copy()
                
        T *= cooling
        
    return best_residues, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    rng = random.Random(42)
    
    best_score = BASE
    best_residues = [0] * NUM_MODULI
    
    # Strategy 1: Greedy with increasing order
    order1 = list(range(NUM_MODULI))
    res1, counts1, score1 = _greedy_init(order1, rng)
    if score1 < best_score:
        best_score = score1
        best_residues = res1[:]
        
    # Strategy 2: Greedy with decreasing order
    order2 = list(range(NUM_MODULI - 1, -1, -1))
    res2, counts2, score2 = _greedy_init(order2, rng)
    if score2 < best_score:
        best_score = score2
        best_residues = res2[:]
        
    # Strategy 3: Random orders + SA
    # We have time for a few SA runs
    # Each SA run should take ~10-20s
    # Let's try 5 runs
    
    runs = 5
    for k in range(runs):
        if time.time() > deadline - 5.0:
            break
            
        # Random order for init
        order = list(range(NUM_MODULI))
        rng.shuffle(order)
        res_init, counts_init, score_init = _greedy_init(order, rng)
        
        # Run SA
        run_deadline = min(deadline - 1.0, time.time() + 15.0)
        res_sa, score_sa = _simulated_annealing(res_init, counts_init, score_init, run_deadline, rng)
        
        if score_sa < best_score:
            best_score = score_sa
            best_residues = res_sa[:]
            
    # Convert residues to classes
    classes = [[ALL_MODULI[i], best_residues[i]] for i in range(NUM_MODULI)]
    
    return classes
