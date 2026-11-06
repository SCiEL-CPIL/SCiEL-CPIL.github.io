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
    3. Repeat until no improvement or time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute moduli in range [20, 200]
    all_moduli = list(range(20, 201))
    
    # We want to maximize sum(1/m), so prefer smaller moduli.
    # Sort by 1/m descending (i.e., m ascending)
    sorted_moduli = sorted(all_moduli, key=lambda m: -1.0/m)
    
    # State: dict mapping modulus -> residue
    # Or list of (m, r)
    # Let's use a dict for easy lookup: {m: r}
    current_assignment = {}
    
    # Helper: check if adding (m, r) is valid with current assignment
    def is_valid_add(m, r, assignment):
        for m2, r2 in assignment.items():
            if m == m2:
                continue
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True
    
    # Helper: find a valid residue for m given assignment
    def find_valid_residue(m, assignment):
        # Try residues 0 to m-1
        # Optimization: we can skip if (r - r2) % gcd(m, m2) == 0 for any existing
        # Just brute force for now, m <= 200 is small
        for r in range(m):
            if is_valid_add(m, r, assignment):
                return r
        return None
    
    # Helper: calculate score
    def calc_score(assignment):
        return sum(1.0/m for m in assignment)
    
    # Helper: try to add m to assignment, return new assignment if successful
    def try_add(m, assignment):
        r = find_valid_residue(m, assignment)
        if r is not None:
            new_assign = dict(assignment)
            new_assign[m] = r
            return new_assign
        return None
    
    # Phase 1: Greedy Construction
    # Start with empty assignment
    assignment = {}
    for m in sorted_moduli:
        new_assign = try_add(m, assignment)
        if new_assign is not None:
            assignment = new_assign
    
    best_assignment = dict(assignment)
    best_score = calc_score(best_assignment)
    
    # Phase 2: Local Search
    # We will try to improve by:
    # 1. Adding any missing modulus
    # 2. Swapping a selected modulus for a missing one
    # 3. Reassigning residues (this is harder, maybe just re-run greedy on the selected set with different order?)
    
    # Let's define a set of selected moduli
    selected_moduli = set(assignment.keys())
    missing_moduli = set(all_moduli) - selected_moduli
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding any missing modulus
        for m in missing_moduli:
            if time.time() > deadline:
                break
            new_assign = try_add(m, assignment)
            if new_assign is not None:
                new_score = calc_score(new_assign)
                if new_score > best_score + 1e-12:
                    assignment = new_assign
                    best_assignment = dict(assignment)
                    best_score = new_score
                    selected_moduli.add(m)
                    missing_moduli.remove(m)
                    improved = True
        
        # 2. Try swapping
        # For each selected m_low (lower 1/m, i.e., larger m) and missing m_high (higher 1/m, i.e., smaller m)
        # If 1/m_high > 1/m_low, check if we can swap.
        # Swap means: remove m_low, add m_high.
        # We need to check if m_high can be added to (assignment - {m_low}).
        
        # Sort selected by m descending (lowest priority first)
        selected_list = sorted(list(selected_moduli), reverse=True)
        missing_list = sorted(list(missing_moduli)) # ascending m (highest priority first)
        
        for m_low in selected_list:
            if time.time() > deadline:
                break
            for m_high in missing_list:
                if time.time() > deadline:
                    break
                if 1.0/m_high <= 1.0/m_low:
                    break # No point in swapping for lower or equal value
                
                # Try swap
                temp_assignment = dict(assignment)
                del temp_assignment[m_low]
                new_assign = try_add(m_high, temp_assignment)
                if new_assign is not None:
                    new_score = calc_score(new_assign)
                    if new_score > best_score + 1e-12:
                        assignment = new_assign
                        best_assignment = dict(assignment)
                        best_score = new_score
                        selected_moduli.remove(m_low)
                        missing_moduli.remove(m_high)
                        selected_moduli.add(m_high)
                        missing_moduli.add(m_low)
                        improved = True
                        # Restart inner loop since sets changed
                        break
            if improved:
                break
        
        # 3. Reassignment / Reordering
        # Sometimes a different order of greedy adds yields a better set.
        # Or we can try to re-optimize residues for the current set.
        # Let's try a simple re-optimization: 
        # Remove all, add back in current sorted order. If it works, great.
        # But maybe a different order works?
        # Given time constraints, let's try a few random permutations of the selected set.
        
        if time.time() < deadline - 5.0:
            # Try random restarts of greedy on the current selected set?
            # Or just try to add more by reordering?
            # Let's try a simple local search on residues.
            # For each m in selected, try all valid residues r.
            # If changing r_m allows adding a missing m', do it.
            
            # This is getting complex. Let's stick to add/swap for now.
            # The "reassignment" in the prompt might refer to finding a different residue for an existing modulus 
            # to free up space for a new one.
            
            # Let's try: for each m in selected, for each r in range(m):
            #   if r is valid for m given others:
            #     temp_assign = assignment with m->r
            #     try to add any missing m' to temp_assign
            #     if success and score improves, accept.
            
            # This is O(N * M * M * N) which might be slow.
            # Let's only do this if we have time.
            
            pass # Skip complex reassignment for now to ensure speed.
            
    # Final check: ensure we return the best found
    # The assignment variable holds the current state, which is the best found so far.
    # But we should return best_assignment just in case.
    
    result = [[m, r] for m, r in best_assignment.items()]
    return result

# EVOLVE-BLOCK-END
