import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    moduli = list(range(20, 201))
    n_moduli = len(moduli)

    # Precompute gcds
    gcds = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # Helper to check if (m_idx, r) is compatible with a set
    def is_compatible(m_idx, r, selected_indices, selected_residues):
        for k in range(len(selected_indices)):
            m_old_idx = selected_indices[k]
            r_old = selected_residues[k]
            g = gcds[m_idx][m_old_idx]
            if (r - r_old) % g == 0:
                return False
        return True

    def get_score(selected_indices):
        return sum(1.0 / moduli[idx] for idx in selected_indices)

    # Initial Solution: Greedy
    # We want to pick small moduli.
    # Sort moduli by value (ascending)
    # For each m, try to find a residue.
    
    def initial_greedy():
        selected_indices = []
        selected_residues = []
        # Try to add in order of increasing m
        for m_idx in range(n_moduli):
            m = moduli[m_idx]
            # Try all residues
            # To make it deterministic but varied, we can shuffle residues
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(m_idx, r, selected_indices, selected_residues):
                    selected_indices.append(m_idx)
                    selected_residues.append(r)
                    break
        return selected_indices, selected_residues

    # Simulated Annealing with deadline
    def run_sa(init_indices, init_residues, deadline):
        current_indices = list(init_indices)
        current_residues = list(init_residues)
        current_score = get_score(current_indices)
        
        best_indices = list(current_indices)
        best_residues = list(current_residues)
        best_score = current_score
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.001
        
        while T > min_T and time.time() < deadline:
            move_type = random.random()
            
            new_indices = list(current_indices)
            new_residues = list(current_residues)
            
            if move_type < 0.4:
                # Add
                if len(current_indices) < n_moduli:
                    added = False
                    # Try to add a random modulus
                    # Bias towards smaller moduli?
                    # Let's pick a random index from the full range, 
                    # but reject if already in set.
                    for _ in range(20):
                        m_idx = random.randint(0, n_moduli - 1)
                        if m_idx not in new_indices:
                            m = moduli[m_idx]
                            residues = list(range(m))
                            random.shuffle(residues)
                            for r in residues:
                                if is_compatible(m_idx, r, new_indices, new_residues):
                                    new_indices.append(m_idx)
                                    new_residues.append(r)
                                    added = True
                                    break
                            if added:
                                break
                    if not added:
                        continue
            elif move_type < 0.8:
                # Change Residue
                if current_indices:
                    pos = random.randint(0, len(current_indices) - 1)
                    m_idx = current_indices[pos]
                    m = moduli[m_idx]
                    old_r = current_residues[pos]
                    
                    temp_indices = new_indices[:pos] + new_indices[pos+1:]
                    temp_residues = new_residues[:pos] + new_residues[pos+1:]
                    
                    residues = list(range(m))
                    # Filter out old_r? Not strictly necessary, but ensures change.
                    # If we pick old_r, delta is 0.
                    random.shuffle(residues)
                    
                    found = False
                    for r in residues:
                        if r == old_r: continue
                        if is_compatible(m_idx, r, temp_indices, temp_residues):
                            new_residues[pos] = r
                            found = True
                            break
                    if not found:
                        continue
            else:
                # Remove
                if current_indices:
                    pos = random.randint(0, len(current_indices) - 1)
                    new_indices.pop(pos)
                    new_residues.pop(pos)
            
            new_score = get_score(new_indices)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_indices = new_indices
                current_residues = new_residues
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_indices = list(current_indices)
                    best_residues = list(current_residues)
            
            T *= cooling_rate
            
        return best_indices, best_residues, best_score

    # Run multiple restarts
    best_overall_score = 0.0
    best_overall_indices = []
    best_overall_residues = []
    
    # We'll do 10 restarts, each with ~10s budget.
    num_restarts = 10
    time_per_restart = 10.0
    
    for i in range(num_restarts):
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        # New initial solution
        init_indices, init_residues = initial_greedy()
        
        res_indices, res_residues, res_score = run_sa(init_indices, init_residues, restart_deadline)
        
        if res_score > best_overall_score:
            best_overall_score = res_score
            best_overall_indices = res_indices
            best_overall_residues = res_residues
            
        if time.time() > deadline:
            break

    # Convert to required format
    result_classes = []
    for i in range(len(best_overall_indices)):
        m_idx = best_overall_indices[i]
        r = best_overall_residues[i]
        m = moduli[m_idx]
        result_classes.append([m, r])
        
    return result_classes
