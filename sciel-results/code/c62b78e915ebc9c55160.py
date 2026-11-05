"""Optimized odd covering system search for moduli dividing 51975."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (evaluator kills at 150s)
    
    base = 51975
    # 51975 = 3^3 * 5^2 * 7 * 11
    # Get all odd divisors of base greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Greedy initialization with vectorized gain computation
    counts = np.zeros(base, dtype=np.int32)
    classes = []
    
    for m in divisors:
        if time.time() > deadline - 5:
            return classes
        
        # Vectorized: compute gain for all residues at once
        # gain[r] = number of positions x ≡ r (mod m) where counts[x] == 0
        uncovered = (counts == 0).astype(np.int32)
        # For each position x, it contributes to residue x % m
        residues = np.arange(base) % m
        gain = np.bincount(residues, weights=uncovered, minlength=m)
        
        best_r = int(np.argmax(gain))
        classes.append([m, best_r])
        counts[best_r::m] += 1
    
    # Local search with vectorized gain computation
    improved = True
    max_passes = 50
    pass_count = 0
    
    while improved and pass_count < max_passes and time.time() < deadline - 10:
        improved = False
        pass_count += 1
        
        indices = list(range(len(classes)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5:
                return classes
            
            m, old_r = classes[idx]
            
            # Temporarily remove this class
            counts[old_r::m] -= 1
            
            # Vectorized: compute gain for all residues
            uncovered = (counts == 0).astype(np.int32)
            residues = np.arange(base) % m
            gain = np.bincount(residues, weights=uncovered, minlength=m)
            
            best_r = int(np.argmax(gain))
            
            if best_r != old_r:
                counts[best_r::m] += 1
                classes[idx][1] = best_r
                improved = True
            else:
                counts[old_r::m] += 1
    
    return classes
# EVOLVE-BLOCK-END
