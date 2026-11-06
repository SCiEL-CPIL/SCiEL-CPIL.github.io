"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Precompute gcd matrix and residue compatibility bitmasks.
    2. Greedy initial construction.
    3. Simulated annealing with multiple move types:
       a. Add a modulus with valid residue
       b. Remove a modulus
       c. Swap a modulus
       d. Reassign a residue
    4. Final polish pass with hill climbing.
    """
    deadline = time.time() + 145.0
    random.seed(42)
    
    # All possible moduli
    all_moduli = list(range(20, 201))
    n_moduli = len(all_moduli)
    mod_idx = {m: i for i, m in enumerate(all_moduli)}
    
    # Precompute gcd between all pairs of moduli
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(n_moduli):
            gcd_table[i][j] = gcd(all_moduli[i], all_moduli[j])
    
    # For each modulus m, precompute for each possible residue r,
    # which other moduli are "blocked" (i.e., have a residue conflict)
    # blocked_mask[i] is a list of length all_moduli[i], where blocked_mask[i][r]
    # is a bitmask of moduli j such that (r % gcd(m_i, m_j)) == (r_j % gcd(m_i, m_j))
    # for SOME r_j in [0, m_j). Actually we need: for a given assignment of residues
    # to other moduli, check if adding (m_i, r) conflicts.
    
    # Better approach: for each modulus i and residue r_i, compute the set of
    # (j, r_j) pairs that conflict. But that's too large.
    
    # Instead, use a constraint satisfaction approach:
    # For a given set of selected moduli with assigned residues, to check if
    # (m_new, r_new) can be added, we check for each existing (m_k, r_k):
    #   (r_new % gcd(m_new, m_k)) != (r_k % gcd(m_new, m_k))
    
    # To speed this up, maintain for each modulus k:
    #   used_residues_mod_g[k] = set of (r_k % g) values for each g
    # Actually, let's just do direct checking but with early termination.
    
    def find_valid_residue(m_idx_new, selected):
        """
        Find a residue r for modulus m_idx_new such that it's compatible with all selected.
        selected: list of (m_idx, r) tuples
        Returns r or None.
        """
        m_new = all_moduli[m_idx_new]
        # For each candidate residue r_new, check against all selected
        for r_new in range(m_new):
            valid = True
            for m_idx_k, r_k in selected:
                g = gcd_table[m_idx_new][m_idx_k]
                if (r_new % g) == (r_k % g):
                    valid = False
                    break
            if valid:
                return r_new
        return None
    
    def try_add_to_selection(selected, m_idx_new):
        """Try to add modulus m_idx_new to selected. Returns (new_selected, r) or (None, None)."""
        r = find_valid_residue(m_idx_new, selected)
        if r is not None:
            new_selected = selected + [(m_idx_new, r)]
            return new_selected, r
        return None, None
    
    def calc_score(selected):
        """Calculate sum of 1/m for the selection."""
        return sum(1.0 / all_moduli[m_idx] for m_idx, r in selected)
    
    # Greedy initial construction
    # Sort by 1/m descending (i.e., m ascending)
    sorted_moduli = sorted(range(n_moduli), key=lambda i: -1.0 / all_moduli[i])
    
    selected = []
    for m_idx in sorted_moduli:
        if time.time() > deadline - 5.0:
            break
        new_sel, r = try_add_to_selection(selected, m_idx)
        if new_sel is not None:
            selected = new_sel
    
    best_selected = selected[:]
    best_score = calc_score(selected)
    
    # Simulated annealing
    current_selected = selected[:]
    current_score = best_score
    
    # Temperature schedule
    t0 = 0.01
    t_end = 1e-6
    n_iterations = 0
    max_iterations = 500000
    
    # Determine how many iterations we can do based on time
    # Each iteration is roughly O(n_selected * m_new) for residue checking
    # Let's estimate ~1000 iterations per second conservatively
    
    start_time = time.time()
    
    for iter_num in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        if time.time() - start_time > 100.0:
            break
            
        # Cool temperature
        progress = iter_num / max_iterations
        temperature = t0 * (t_end / t0) ** progress
        
        # Choose a random move
        move_type = random.random()
        n_sel = len(current_selected)
        
        if move_type < 0.3 and n_sel < n_moduli:
            # Add a random missing modulus
            current_moduli_set = set(m_idx for m_idx, r in current_selected)
            missing = [i for i in range(n_moduli) if i not in current_moduli_set]
            if not missing:
                continue
            m_idx_new = random.choice(missing)
            r_new = find_valid_residue(m_idx_new, current_selected)
            if r_new is None:
                continue
            new_selected = current_selected + [(m_idx_new, r_new)]
            new_score = calc_score(new_selected)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_selected = new_selected
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = current_selected[:]
        
        elif move_type < 0.5 and n_sel > 0:
            # Remove a random modulus
            idx_to_remove = random.randint(0, n_sel - 1)
            new_selected = current_selected[:idx_to_remove] + current_selected[idx_to_remove+1:]
            new_score = calc_score(new_selected)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_selected = new_selected
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = current_selected[:]
        
        elif move_type < 0.8 and n_sel > 0:
            # Swap: remove one, add another
            current_moduli_set = set(m_idx for m_idx, r in current_selected)
            missing = [i for i in range(n_moduli) if i not in current_moduli_set]
            if not missing:
                continue
            idx_to_remove = random.randint(0, n_sel - 1)
            m_idx_add = random.choice(missing)
            
            # Temporarily remove
            temp_selected = current_selected[:idx_to_remove] + current_selected[idx_to_remove+1:]
            r_new = find_valid_residue(m_idx_add, temp_selected)
            if r_new is None:
                continue
            new_selected = temp_selected + [(m_idx_add, r_new)]
            new_score = calc_score(new_selected)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_selected = new_selected
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = current_selected[:]
        
        else:
            # Reassign a residue for a random selected modulus
            if n_sel == 0:
                continue
            idx = random.randint(0, n_sel - 1)
            m_idx, r_old = current_selected[idx]
            m_val = all_moduli[m_idx]
            
            # Try a random new residue
            r_new = random.randint(0, m_val - 1)
            if r_new == r_old:
                continue
            
            # Check if r_new is valid with all other selected
            valid = True
            for j, (m_idx_j, r_j) in enumerate(current_selected):
                if j == idx:
                    continue
                g = gcd_table[m_idx][m_idx_j]
                if (r_new % g) == (r_j % g):
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_selected = current_selected[:]
            new_selected[idx] = (m_idx, r_new)
            # Score doesn't change with residue reassignment
            delta = 0.0
            # Always accept residue reassignment (it doesn't change score but may enable future moves)
            current_selected = new_selected
            current_score = calc_score(current_selected)  # Should be same
            if current_score > best_score:
                best_score = current_score
                best_selected = current_selected[:]
        
        n_iterations += 1
    
    # Final polish: hill climbing from best solution
    current_selected = best_selected[:]
    current_score = best_score
    
    polish_deadline = deadline - 2.0
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        n_sel = len(current_selected)
        current_moduli_set = set(m_idx for m_idx, r in current_selected)
        missing = [i for i in range(n_moduli) if i not in current_moduli_set]
        
        # Try adding
        for m_idx_add in missing:
            if time.time() > polish_deadline:
                break
            r_new = find_valid_residue(m_idx_add, current_selected)
            if r_new is not None:
                new_selected = current_selected + [(m_idx_add, r_new)]
                new_score = calc_score(new_selected)
                if new_score > current_score + 1e-12:
                    current_selected = new_selected
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = current_selected[:]
                    improved = True
        
        # Try swaps
        current_moduli_set = set(m_idx for m_idx, r in current_selected)
        missing = [i for i in range(n_moduli) if i not in current_moduli_set]
        for m_idx_add in missing:
            if time.time() > polish_deadline:
                break
            for i in range(len(current_selected)):
                if time.time() > polish_deadline:
                    break
                m_idx_remove, r_remove = current_selected[i]
                if all_moduli[m_idx_add] >= all_moduli[m_idx_remove]:
                    continue  # Only swap if adding smaller modulus
                temp_selected = current_selected[:i] + current_selected[i+1:]
                r_new = find_valid_residue(m_idx_add, temp_selected)
                if r_new is not None:
                    new_selected = temp_selected + [(m_idx_add, r_new)]
                    new_score = calc_score(new_selected)
                    if new_score > current_score + 1e-12:
                        current_selected = new_selected
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = current_selected[:]
                        improved = True
                        break
        
        # Try residue reassignments to enable additions
        for i in range(len(current_selected)):
            if time.time() > polish_deadline:
                break
            m_idx, r_old = current_selected[i]
            m_val = all_moduli[m_idx]
            for r_new in range(m_val):
                if r_new == r_old:
                    continue
                valid = True
                for j, (m_idx_j, r_j) in enumerate(current_selected):
                    if j == i:
                        continue
                    g = gcd_table[m_idx][m_idx_j]
                    if (r_new % g) == (r_j % g):
                        valid = False
                        break
                if not valid:
                    continue
                old_sel = current_selected[:]
                current_selected[i] = (m_idx, r_new)
                # Try to add a missing modulus
                current_moduli_set = set(m_idx2 for m_idx2, r2 in current_selected)
                for m_idx_add in range(n_moduli):
                    if m_idx_add in current_moduli_set:
                        continue
                    r_add = find_valid_residue(m_idx_add, current_selected)
                    if r_add is not None:
                        new_selected = current_selected + [(m_idx_add, r_add)]
                        new_score = calc_score(new_selected)
                        if new_score > current_score + 1e-12:
                            current_selected = new_selected
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_selected = current_selected[:]
                            improved = True
                            break
                if not improved:
                    current_selected = old_sel
    
    # Convert back to required format [[m, r], ...]
    result = [[all_moduli[m_idx], r] for m_idx, r in best_selected]
    return result
# EVOLVE-BLOCK-END
