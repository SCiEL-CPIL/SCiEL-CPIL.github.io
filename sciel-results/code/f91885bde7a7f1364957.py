import time
import random
import math
import numpy as np

N = 51975

def get_divisors(n):
    divs = []
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.append(n)
    return sorted(divs)

MODULI = get_divisors(N)
K = len(MODULI)

def initial_coverage(residues):
    counts = np.zeros(N, dtype=np.int32)
    for m, r in zip(MODULI, residues):
        counts[r::m] += 1
    score = int(np.sum(counts == 0))
    return counts, score

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Phase 1: Greedy construction with descending moduli order
    # This establishes a sparse baseline where large moduli cover few residues,
    # leaving many gaps for smaller moduli to fill efficiently.
    covered = np.zeros(N, dtype=np.uint8)
    greedy_residues = [0] * K
    greedy_classes = []
    
    # Sort indices by modulus descending
    indices_desc = sorted(range(K), key=lambda i: MODULI[i], reverse=True)
    
    for idx in indices_desc:
        m = MODULI[idx]
        best_r = 0
        best_gain = -1
        # For each residue r, compute how many new residues it covers
        # We can do this efficiently with numpy
        # gain[r] = sum(1 - covered[r::m])
        # But iterating m times and summing is O(N) per r, so O(N*m) total which is too slow for large m.
        # Wait, N=51975, max m=51975.
        # Summing N/m elements for m residues is N total operations per modulus.
        # Total for all moduli is sum(N/m) which is manageable.
        
        # Actually, let's just loop r and use numpy sum for speed.
        for r in range(m):
            # Only compute if there's potential gain
            # Quick check: if covered[r::m] is all 1s, gain is 0.
            # np.sum(1 - covered[r::m])
            gain = np.sum(1 - covered[r::m])
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        greedy_residues[idx] = best_r
        covered[best_r::m] = 1
        greedy_classes.append([m, best_r])
    
    counts, current_score = initial_coverage(greedy_residues)
    
    best_residues = greedy_residues[:]
    best_score = current_score
    
    # Phase 2: Simulated Annealing / Local Search
    # Start from the greedy solution
    current_residues = greedy_residues[:]
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-5
    
    # Precompute step sizes for each modulus? No, just random.
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, K - 1)
        m = MODULI[i]
        
        old_r = current_residues[i]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
            
        delta = 0
        
        # Process S_old (removing coverage)
        # We need to update counts and score
        # For x in range(old_r, N, m):
        #   if counts[x] == 1: delta += 1
        #   counts[x] -= 1
        
        # Vectorized update is tricky for the score delta because we need to know
        # how many went from 1 to 0.
        # old_slice = counts[old_r::m]
        # delta += np.sum(old_slice == 1)
        # counts[old_r::m] -= 1
        
        # new_slice = counts[new_r::m]
        # delta -= np.sum(new_slice == 0)
        # counts[new_r::m] += 1
        
        # This is much faster than Python loops.
        
        old_slice = counts[old_r::m]
        delta += np.sum(old_slice == 1)
        counts[old_r::m] -= 1
        
        new_slice = counts[new_r::m]
        delta -= np.sum(new_slice == 0)
        counts[new_r::m] += 1
        
        new_score = current_score + delta
        
        # Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_residues[i] = new_r
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            # Revert
            # counts[old_r::m] += 1
            # counts[new_r::m] -= 1
            counts[old_r::m] += 1
            counts[new_r::m] -= 1
            # current_residues[i] is already old_r
            
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Occasionally restart or perturb if stuck?
        # For now, standard SA.
        
        # Check time inside loop to be safe
        if (time.time() - start_time) > time_limit - 5:
            break
            
    classes = [[m, r] for m, r in zip(MODULI, best_residues)]
    return classes
