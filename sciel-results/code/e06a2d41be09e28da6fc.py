"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Moduli range
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # For each modulus m, the possible residues are 0..m-1
    # But we need to find a valid assignment of residues such that
    # for any pair (m_i, r_i), (m_j, r_j): r_i != r_j mod gcd(m_i, m_j)
    
    # Strategy: Greedy + Local Search
    # 1. Start with empty set
    # 2. Try to add moduli in order of 1/m (descending, i.e., small m first)
    # 3. For each modulus, find a valid residue
    # 4. Then do local search: try swaps, additions, reassignments
    
    def find_valid_residue(m, selected):
        """Find a residue r in [0, m) such that r != r_j mod gcd(m, m_j) for all selected (m_j, r_j)."""
        for r in range(m):
            valid = True
            for m_j, r_j in selected:
                g = gcd(m, m_j)
                if r % g == r_j % g:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def is_valid_assignment(selected):
        """Check if the current assignment is valid."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                m_i, r_i = selected[i]
                m_j, r_j = selected[j]
                g = gcd(m_i, m_j)
                if r_i % g == r_j % g:
                    return False
        return True
    
    def calc_score(selected):
        """Calculate the score as sum of 1/m."""
        return sum(Fraction(1, m) for m, r in selected)
    
    # Greedy initial construction
    selected = []
    # Sort moduli by 1/m descending (small m first)
    sorted_moduli = sorted(moduli, key=lambda m: -1.0/m)
    
    for m in sorted_moduli:
        if time.time() > deadline - 5:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
    
    best_selected = [tuple(x) for x in selected]
    best_score = calc_score(best_selected)
    
    # Local search phase
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations:
        if time.time() > deadline - 5:
            break
        improved = False
        iteration += 1
        
        # Try adding missing moduli
        current_moduli = set(m for m, r in best_selected)
        for m in sorted_moduli:
            if m in current_moduli:
                continue
            r = find_valid_residue(m, best_selected)
            if r is not None:
                new_selected = best_selected + [(m, r)]
                new_score = calc_score(new_selected)
                if new_score > best_score:
                    best_selected = new_selected
                    best_score = new_score
                    improved = True
        
        # Try swaps: remove a selected modulus and add a missing one
        current_moduli = set(m for m, r in best_selected)
        for m_remove in [m for m, r in best_selected]:
            if time.time() > deadline - 5:
                break
            # Try removing m_remove
            remaining = [(m, r) for m, r in best_selected if m != m_remove]
            for m_add in sorted_moduli:
                if m_add in current_moduli or m_add == m_remove:
                    continue
                if 1.0/m_add <= 1.0/m_remove:
                    continue  # Only consider swaps that increase score
                r_add = find_valid_residue(m_add, remaining)
                if r_add is not None:
                    new_selected = remaining + [(m_add, r_add)]
                    new_score = calc_score(new_selected)
                    if new_score > best_score:
                        best_selected = new_selected
                        best_score = new_score
                        improved = True
                        current_moduli = set(m for m, r in best_selected)
        
        # Try reassigning residues for selected moduli to allow more additions
        for i, (m_i, r_i) in enumerate(best_selected):
            if time.time() > deadline - 5:
                break
            # Try changing residue of m_i
            for r_new in range(m_i):
                if r_new == r_i:
                    continue
                # Check if new residue is valid with all other selected
                valid = True
                for j, (m_j, r_j) in enumerate(best_selected):
                    if i == j:
                        continue
                    g = gcd(m_i, m_j)
                    if r_new % g == r_j % g:
                        valid = False
                        break
                if valid:
                    # Try this new assignment and see if we can add more
                    trial = list(best_selected)
                    trial[i] = (m_i, r_new)
                    # Try to add missing moduli with this new assignment
                    trial_moduli = set(m for m, r in trial)
                    for m_add in sorted_moduli:
                        if m_add in trial_moduli:
                            continue
                        r_add = find_valid_residue(m_add, trial)
                        if r_add is not None:
                            trial.append((m_add, r_add))
                            trial_moduli.add(m_add)
                    new_score = calc_score(trial)
                    if new_score > best_score:
                        best_selected = trial
                        best_score = new_score
                        improved = True
    
    # Convert to required format
    result = [[m, r] for m, r in best_selected]
    return result
# EVOLVE-BLOCK-END
