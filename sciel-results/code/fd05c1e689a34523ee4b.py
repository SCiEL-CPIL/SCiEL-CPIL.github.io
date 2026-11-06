"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction: Iterate moduli in order of 1/m (descending), adding a modulus if a valid residue exists.
    2. Local Search (Hill Climbing): Try to improve the set of moduli by:
       a. Adding a missing modulus if a valid residue can be found.
       b. Swapping a selected modulus (with lower 1/m) for a missing modulus (with higher 1/m) if the swap results in a valid assignment and higher total score.
       c. Reassigning residues to allow more moduli to be added.
    3. Return the best solution found within the time budget.
    """
    deadline = time.time() + 120  # Safety margin before 150s limit
    
    def is_valid(m, r, existing):
        """Check if (m, r) is disjoint from all existing (m_i, r_i)."""
        for m_i, r_i in existing:
            g = gcd(m, m_i)
            if r % g == r_i % g:
                return False
        return True
    
    def find_valid_residue(m, existing):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all existing."""
        for r in range(m):
            if is_valid(m, r, existing):
                return r
        return None
    
    def calc_score(existing):
        """Calculate the score as sum of 1/m."""
        return sum(1.0 / m for m, _ in existing)
    
    def try_add(existing, m):
        """Try to add modulus m with a valid residue."""
        r = find_valid_residue(m, existing)
        if r is not None:
            return existing + [(m, r)], True
        return existing, False
    
    # Step 1: Greedy initial construction
    # Sort moduli by 1/m descending (i.e., m ascending)
    moduli = list(range(20, 201))
    moduli.sort(key=lambda m: -1.0/m)  # Higher 1/m first
    
    solution = []
    for m in moduli:
        if time.time() > deadline - 10:
            break
        new_sol, ok = try_add(solution, m)
        if ok:
            solution = new_sol
    
    best_score = calc_score(solution)
    best_solution = list(solution)
    
    # Step 2: Local search
    # Try adding missing moduli
    selected_moduli = set(m for m, _ in best_solution)
    missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try adding each missing modulus
        for m in missing_moduli:
            if time.time() > deadline - 5:
                break
            new_sol, ok = try_add(best_solution, m)
            if ok:
                new_score = calc_score(new_sol)
                if new_score > best_score + 1e-12:
                    best_solution = new_sol
                    best_score = new_score
                    improved = True
                    # Update missing moduli
                    selected_moduli = set(m2 for m2, _ in best_solution)
                    missing_moduli = [m2 for m2 in range(20, 201) if m2 not in selected_moduli]
        
        # Try swapping: remove a low-value modulus, add a high-value missing one
        if time.time() > deadline - 5:
            break
        
        # Sort selected by 1/m ascending (lowest value first)
        selected_sorted = sorted(best_solution, key=lambda x: 1.0/x[0])
        
        for i, (m_remove, r_remove) in enumerate(selected_sorted):
            if time.time() > deadline - 5:
                break
            
            # Try removing this modulus and adding a missing one
            without_remove = [x for j, x in enumerate(best_solution) if x[0] != m_remove]
            
            for m_add in missing_moduli:
                if 1.0/m_add <= 1.0/m_remove:
                    continue  # Only consider swaps that improve score
                
                new_sol, ok = try_add(without_remove, m_add)
                if ok:
                    new_score = calc_score(new_sol)
                    if new_score > best_score + 1e-12:
                        best_solution = new_sol
                        best_score = new_score
                        improved = True
                        # Update sets
                        selected_moduli = set(m2 for m2, _ in best_solution)
                        missing_moduli = [m2 for m2 in range(20, 201) if m2 not in selected_moduli]
                        break
    
    # Convert to required format [[m, r], ...]
    result = [[m, r] for m, r in best_solution]
    return result

# EVOLVE-BLOCK-END
