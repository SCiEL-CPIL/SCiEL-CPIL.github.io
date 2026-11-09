# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    conflict = [[False] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            if g == 1:
                conflict[i][j] = True
                conflict[j][i] = True
                
    def is_valid_addition(indices, new_idx):
        """Check if new_idx can be added to indices without coprime conflict."""
        for i in indices:
            if conflict[i][new_idx]:
                return False
        return True

    def solve_residues_fast(indices):
        """
        Fast greedy residue assignment.
        Returns (success, residues_dict).
        """
        if not indices:
            return True, {}
        
        # Sort indices by modulus descending (larger first) to leave more options?
        # Or ascending?
        # Let's try ascending (smaller first) as they are more constrained.
        sorted_indices = sorted(indices, key=lambda i: moduli[i])
        
        residues = {}
        # Precompute masks for each modulus?
        # We need to compute forbidden mask for each new m based on existing residues.
        
        for idx in sorted_indices:
            m = moduli[idx]
            forbidden_mask = 0
            for idx2, r2 in residues.items():
                g = gcd_table[idx][idx2]
                r0 = r2 % g
                # Construct mask for forbidden residues in [0, m)
                # r0, r0+g, r0+2g ...
                mask = 0
                for r in range(r0, m, g):
                    mask |= (1 << r)
                forbidden_mask |= mask
            
            # Find a valid r
            valid_mask = ((1 << m) - 1) & ~forbidden_mask
            if valid_mask == 0:
                return False, {}
            
            # Pick a random valid r
            # Extract bits
            r = 0
            # Find first set bit? Or random?
            # Random is better for diversity, but we want speed.
            # Let's pick the first one for determinism in the fast path?
            # No, let's pick a random one from the set bits.
            # To be fast, just pick the lowest set bit.
            while not (valid_mask & 1):
                valid_mask >>= 1
                r += 1
            
            residues[idx] = r
        
        return True, residues

    def score_moduli(indices):
        """Calculate score for a set of moduli indices."""
        if not indices:
            return 0.0
        
        # Fast validity check is done by SA moves, but double check?
        # SA ensures validity.
        
        # Try to solve residues
        success, residues = solve_residues_fast(indices)
        if not success:
            return 0.0
        
        # Calculate sum
        total = 0.0
        for idx in indices:
            total += 1.0 / moduli[idx]
        return total

    # Simulated Annealing
    # Initial state: All even moduli
    initial_indices = [i for i, m in enumerate(moduli) if m % 2 == 0]
    current_indices = initial_indices
    current_score = score_moduli(current_indices)
    
    best_indices = current_indices[:]
    best_score = current_score
    
    T = 0.5
    T_min = 0.001
    alpha = 0.995
    
    step = 0
    while time.time() < deadline - 10.0:
        # Generate neighbor
        neighbor = current_indices[:]
        move_type = random.randint(0, 2)
        
        if move_type == 0: # Add
            # Pick a random modulus not in current
            # To speed up, maintain a set of unused?
            # For now, list comprehension
            candidates = [i for i in range(n_mod) if i not in neighbor]
            if candidates:
                i_add = random.choice(candidates)
                if is_valid_addition(neighbor, i_add):
                    neighbor.append(i_add)
                else:
                    # Invalid move, skip evaluation or treat as bad?
                    # Better to skip and generate another move?
                    # Or just evaluate as 0?
                    # Let's skip to avoid wasting time on invalid states.
                    continue
        elif move_type == 1: # Remove
            if neighbor:
                i_rem = random.choice(neighbor)
                neighbor.remove(i_rem)
        else: # Swap
            if neighbor:
                i_rem = random.choice(neighbor)
                candidates = [i for i in range(n_mod) if i not in neighbor]
                if candidates:
                    i_add = random.choice(candidates)
                    if is_valid_addition([x for x in neighbor if x != i_rem], i_add):
                        neighbor.remove(i_rem)
                        neighbor.append(i_add)
                    else:
                        continue
        
        # Evaluate neighbor
        neighbor_score = score_moduli(neighbor)
        
        delta = neighbor_score - current_score
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            current_indices = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_indices = neighbor[:]
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        step += 1

    # Final polish
    best_moduli = [moduli[i] for i in best_indices]
    success, residues = solve_residues_fast(best_indices)
    
    if not success:
        # Fallback
        return []
        
    classes = []
    for idx in best_indices:
        m = moduli[idx]
        classes.append([m, residues[idx]])
        
    return classes
# EVOLVE-BLOCK-END
