"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _all_odd_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    base = 51975
    # Precompute all odd divisors > 1 of base
    divisors = _all_odd_divisors(base)
    
    best_score = base  # initially all uncovered
    best_classes = None
    rng = random.Random(42)
    
    moduli = divisors
    num_moduli = len(moduli)
    
    # Sort moduli: smaller moduli first (they cover more residues)
    moduli_sorted = sorted(moduli)
    
    # Multiple restarts with different greedy orders and SA
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different order for each restart
        if restart == 0:
            order = moduli_sorted  # ascending
        elif restart == 1:
            order = list(reversed(moduli_sorted))  # descending
        elif restart == 2:
            order = list(moduli_sorted)
            rng.shuffle(order)
        elif restart == 3:
            # Priority: moduli that are "critical" (small and prime-like)
            order = sorted(moduli_sorted, key=lambda m: (-1.0/m, m))
        else:
            order = list(moduli_sorted)
            rng.shuffle(order)
        
        # Greedy initialization using counts
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli
        # Map from modulus to index in residues array
        mod_to_idx = {m: i for i, m in enumerate(moduli)}
        
        for m in order:
            idx = mod_to_idx[m]
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[idx] = best_r
            counts[best_r::m] += 1
        
        current_score = int(np.count_nonzero(counts == 0))
        if current_score < best_score:
            best_score = current_score
            best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
        
        if time.time() > deadline:
            break
        
        # Simulated Annealing on this initialization
        temperature = 3.0 + restart * 0.5
        cooling = 0.997
        min_temp = 0.05
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline and temperature > min_temp:
            i = rng.randint(0, num_moduli - 1)
            m = moduli[i]
            old_r = residues[i]
            new_r = rng.randint(0, m - 1)
            if new_r == old_r:
                continue
            
            # Update counts
            counts[old_r::m] -= 1
            counts[new_r::m] += 1
            
            new_score = int(np.count_nonzero(counts == 0))
            delta = new_score - current_score
            
            if delta < 0 or (delta == 0 and rng.random() < 0.1) or rng.random() < np.exp(-delta / temperature):
                residues[i] = new_r
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [[moduli[j], residues[j]] for j in range(num_moduli)]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                counts[new_r::m] -= 1
                counts[old_r::m] += 1
                no_improve_count += 1
            
            temperature *= cooling
            
            # Early stop if stuck
            if no_improve_count > max_no_improve and temperature < 0.5:
                break
        
        # Local search refinement after SA
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(num_moduli):
                    m = moduli[i]
                    old_r = residues[i]
                    best_r = old_r
                    best_local_score = current_score
                    
                    # Try all residues for this modulus
                    for r in range(m):
                        if r == old_r:
                            continue
                        counts[old_r::m] -= 1
                        counts[r::m] += 1
                        local_score = int(np.count_nonzero(counts == 0))
                        counts[r::m] -= 1
                        counts[old_r::m] += 1
                        
                        if local_score < best_local_score:
                            best_local_score = local_score
                            best_r = r
                    
                    if best_r != old_r:
                        counts[old_r::m] -= 1
                        counts[best_r::m] += 1
                        residues[i] = best_r
                        current_score = best_local_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_classes = [[moduli[j], residues[j]] for j in range(num_moduli)]
    
    return best_classes
# EVOLVE-BLOCK-END
