import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    base = 51975
    # Generate odd divisors > 1
    moduli = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    best_uncovered = float('inf')
    best_residues = None
    
    # Try multiple random seeds for diversity
    for seed in range(20):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Greedy initialization with random ordering
        count_cover = np.zeros(base, dtype=np.int8)
        residues = {}
        
        # Shuffle moduli for diversity
        shuffled_moduli = moduli.copy()
        random.shuffle(shuffled_moduli)
        
        # Process moduli: small ones first tend to cover more
        # But let's try a mix: process by size with some randomness
        sorted_moduli = sorted(shuffled_moduli, key=lambda m: (m, random.random()))
        
        for m in sorted_moduli:
            # Find r that maximizes coverage of currently uncovered (count_cover==0)
            best_r = 0
            best_score = -1
            
            # Sample residues for large moduli to save time
            if m > 10000 and time.time() > deadline - 5:
                # Sample a subset
                sample_size = min(m, 500)
                samples = random.sample(range(m), sample_size)
                for r in samples:
                    s = np.sum(count_cover[r::m] == 0)
                    if s > best_score:
                        best_score = s
                        best_r = r
            else:
                for r in range(m):
                    s = np.sum(count_cover[r::m] == 0)
                    if s > best_score:
                        best_score = s
                        best_r = r
            
            residues[m] = best_r
            count_cover[best_r::m] += 1
        
        # Local search with simulated annealing
        uncovered = np.sum(count_cover == 0)
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_residues = residues.copy()
        
        # Simulated annealing local search
        current_uncovered = uncovered
        temperature = 50.0
        cooling = 0.995
        min_temperature = 0.1
        
        while temperature > min_temperature and time.time() < deadline - 5:
            # Pick a random modulus
            m = random.choice(moduli)
            r_old = residues[m]
            
            # Pick a new random residue
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
            
            # Calculate delta in uncovered count
            # Remove old class, add new class
            count_cover[r_old::m] -= 1
            old_uncovered = np.sum(count_cover == 0)
            
            count_cover[r_new::m] += 1
            new_uncovered = np.sum(count_cover == 0)
            
            delta = new_uncovered - old_uncovered
            
            # Accept if improvement, or with probability exp(-delta/temperature)
            if delta <= 0 or (delta > 0 and random.random() < np.exp(-delta / temperature)):
                residues[m] = r_new
                current_uncovered = new_uncovered
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues.copy()
            else:
                # Revert
                count_cover[r_new::m] -= 1
                count_cover[r_old::m] += 1
            
            temperature *= cooling
        
        # Final local search pass (deterministic)
        for m in moduli:
            if time.time() > deadline - 2:
                break
                
            r_old = residues[m]
            count_others = count_cover.copy()
            count_others[r_old::m] -= 1
            
            best_r = r_old
            best_score = np.sum(count_others[r_old::m] == 0)
            
            # Check a few random residues plus the current best
            if m > 5000:
                candidates = [r_old] + random.sample(range(m), min(100, m - 1))
            else:
                candidates = list(range(m))
            
            for r in candidates:
                if r == r_old:
                    continue
                s = np.sum(count_others[r::m] == 0)
                if s > best_score:
                    best_score = s
                    best_r = r
            
            if best_r != r_old:
                count_cover[r_old::m] -= 1
                count_cover[best_r::m] += 1
                residues[m] = best_r
        
        # Update best
        final_uncovered = np.sum(count_cover == 0)
        if final_uncovered < best_uncovered:
            best_uncovered = final_uncovered
            best_residues = residues.copy()
    
    # Construct return value from best solution
    if best_residues is None:
        # Fallback: use the last residues
        best_residues = residues
    
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
