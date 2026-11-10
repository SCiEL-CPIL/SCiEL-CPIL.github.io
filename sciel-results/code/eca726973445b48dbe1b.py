import numpy as np
import random
import time

def construct():
    base = 51975
    
    # Generate divisors > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Map divisor to index in list for quick access
    div_set = set(divisors)
    
    # Initial residues: all 0
    residues = {m: 0 for m in divisors}
    
    # counts[x] = number of classes covering x
    counts = np.zeros(base, dtype=np.int8)
    
    # Initialize counts
    for m in divisors:
        r = residues[m]
        indices = np.arange(r, base, m)
        counts[indices] += 1
        
    current_uncovered = np.sum(counts == 0)
    best_uncovered = current_uncovered
    best_residues = residues.copy()
    
    start_time = time.time()
    time_limit = 110
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random modulus
        m = random.choice(divisors)
        old_r = residues[m]
        
        # Pick a new random residue
        new_r = random.randint(0, m - 1)
        if new_r == old_r:
            continue
            
        # Calculate delta in uncovered count
        # Residues affected:
        # Old coverage: indices_old = np.arange(old_r, base, m)
        # New coverage: indices_new = np.arange(new_r, base, m)
        
        # We need to know how many residues are covered ONLY by this class in old state
        # and how many are uncovered in old state but covered in new state.
        
        # Efficient way:
        # 1. Remove old coverage from counts (temporarily)
        indices_old = np.arange(old_r, base, m)
        counts[indices_old] -= 1
        
        # 2. Add new coverage
        indices_new = np.arange(new_r, base, m)
        counts[indices_new] += 1
        
        # 3. Calculate new uncovered
        # Instead of summing whole array, we can track changes?
        # Summing 51975 elements is fast (~0.1ms). 
        # Let's just sum.
        new_uncovered = np.sum(counts == 0)
        
        delta = new_uncovered - current_uncovered
        
        if delta < 0:
            # Improvement
            current_uncovered = new_uncovered
            residues[m] = new_r
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues.copy()
        else:
            # Acceptance probability
            if T > 0.01:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_uncovered = new_uncovered
                    residues[m] = new_r
                else:
                    # Revert
                    counts[indices_new] -= 1
                    counts[indices_old] += 1
            else:
                # Revert
                counts[indices_new] -= 1
                counts[indices_old] += 1
        
        T *= cooling_rate
        if T < 0.001:
            T = 1.0 # Restart annealing? Or just keep it low for hill climbing.
            # Let's restart to escape local minima
            T = 1.0

    # Construct return value
    # We need to return [[m, r], ...]
    # Using best_residues
    result = [[m, best_residues[m]] for m in divisors]
    
    # Optional: Verify best score? Not needed, we track it.
    # But wait, if we restart T, we might lose the "best" tracking if we don't update it correctly.
    # The code above updates best_uncovered only on improvement (delta < 0).
    # This is correct for Hill Climbing. For SA, we might want to track the best state ever seen.
    
    return result
