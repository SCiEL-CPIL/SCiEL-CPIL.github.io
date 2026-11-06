"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Fast greedy construction as initial solution
    2. Backtracking search with constraint propagation to find optimal assignment
    3. Local search with residue reassignment to improve
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    all_moduli = list(range(20, 201))
    
    # Precompute gcd pairs
    gcd_cache = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_cache[(m1, m2)] = gcd(m1, m2)
    
    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        key = (min(m1, m2), max(m1, m2))
        if key not in gcd_cache:
            gcd_cache[key] = gcd(m1, m2)
        return gcd_cache[key]
    
    def calc_score(assignment):
        return sum(1.0/m for m in assignment)
    
    def is_compatible(m, r, assignment, exclude=None):
        """Check if (m, r) is compatible with all in assignment (except exclude)."""
        for m2, r2 in assignment.items():
            if m2 == exclude:
                continue
            if m == m2:
                continue
            g = get_gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_valid_residues(m, assignment, exclude=None):
        """Find all valid residues for m given assignment."""
        valid = []
        for r in range(m):
            if is_compatible(m, r, assignment, exclude):
                valid.append(r)
        return valid
    
    def greedy_construct(moduli_order, assignment=None):
        """Greedy construction with given order."""
        if assignment is None:
            assignment = {}
        for m in moduli_order:
            if m in assignment:
                continue
            residues = find_valid_residues(m, assignment)
            if residues:
                # Pick residue that maximizes future flexibility
                # Simple heuristic: pick middle residue
                assignment[m] = residues[len(residues)//2]
        return assignment
    
    # Phase 1: Initial greedy solution
    # Sort by 1/m descending (m ascending)
    initial_order = sorted(all_moduli, key=lambda m: m)
    best_assignment = greedy_construct(initial_order)
    best_score = calc_score(best_assignment)
    
    # Phase 2: Backtracking search with constraint propagation
    # We'll try to add more moduli or swap for better ones
    # Use a priority queue approach: try to add smallest missing moduli first
    
    def backtrack_search(assignment, candidate_moduli, max_nodes=50000):
        """
        Backtracking search to improve assignment.
        candidate_moduli: list of moduli we might want to add (sorted by priority)
        Returns improved assignment or None.
        """
        nodes_visited = 0
        best_local = dict(assignment)
        best_local_score = calc_score(best_local)
        
        def search(current_assign, remaining_candidates, depth=0):
            nonlocal nodes_visited, best_local, best_local_score
            
            if nodes_visited > max_nodes:
                return
            nodes_visited += 1
            
            if time.time() > deadline:
                return
            
            if not remaining_candidates:
                return
            
            # Try to add the first candidate
            m = remaining_candidates[0]
            rest = remaining_candidates[1:]
            
            residues = find_valid_residues(m, current_assign)
            if residues:
                # Try a few residues (not all, to limit branching)
                sample_size = min(len(residues), 3)
                if len(residues) <= 3:
                    sample = residues
                else:
                    # Pick residues that are "spread out"
                    sample = [residues[0], residues[len(residues)//2], residues[-1]]
                
                for r in sample:
                    new_assign = dict(current_assign)
                    new_assign[m] = r
                    new_score = calc_score(new_assign)
                    if new_score > best_local_score + 1e-12:
                        best_local = dict(new_assign)
                        best_local_score = new_score
                    
                    # Recurse with remaining candidates
                    search(new_assign, rest, depth + 1)
            
            # Also try skipping this candidate and trying next
            if rest:
                search(current_assign, rest, depth)
        
        # Limit candidates to top 20 by 1/m value
        top_candidates = sorted(candidate_moduli, key=lambda m: m)[:20]
        search(assignment, top_candidates)
        
        return best_local
    
    # Phase 3: Local search with swaps and additions
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 5.0:
        improved = False
        iteration += 1
        
        selected = set(best_assignment.keys())
        missing = [m for m in all_moduli if m not in selected]
        
        # Try to add missing moduli
        for m in sorted(missing, key=lambda x: x):
            if time.time() > deadline:
                break
            residues = find_valid_residues(m, best_assignment)
            if residues:
                # Pick best residue
                best_r = residues[0]
                for r in residues:
                    # Check if this residue allows more future additions
                    # Simple: just pick first valid
                    best_r = r
                    break
                
                new_assign = dict(best_assignment)
                new_assign[m] = best_r
                new_score = calc_score(new_assign)
                if new_score > best_score + 1e-12:
                    best_assignment = new_assign
                    best_score = new_score
                    improved = True
        
        # Try swaps: remove large m, add small m
        selected_list = sorted(selected, reverse=True)  # Largest first (lowest priority)
        missing_list = sorted(missing)  # Smallest first (highest priority)
        
        for m_remove in selected_list:
            if time.time() > deadline:
                break
            for m_add in missing_list:
                if time.time() > deadline:
                    break
                if 1.0/m_add <= 1.0/m_remove:
                    break
                
                # Try removing m_remove and adding m_add
                temp_assign = {k: v for k, v in best_assignment.items() if k != m_remove}
                residues = find_valid_residues(m_add, temp_assign)
                if residues:
                    new_assign = dict(temp_assign)
                    new_assign[m_add] = residues[0]
                    new_score = calc_score(new_assign)
                    if new_score > best_score + 1e-12:
                        best_assignment = new_assign
                        best_score = new_score
                        improved = True
                        break
            if improved:
                break
        
        # Try residue reassignment to enable more additions
        if time.time() < deadline - 10.0 and not improved:
            # For each selected modulus, try different residues to see if it enables additions
            for m in list(best_assignment.keys()):
                if time.time() > deadline:
                    break
                
                current_r = best_assignment[m]
                other_assign = {k: v for k, v in best_assignment.items() if k != m}
                residues = find_valid_residues(m, other_assign)
                
                for r in residues:
                    if r == current_r:
                        continue
                    
                    new_assign = dict(other_assign)
                    new_assign[m] = r
                    
                    # Try to add any missing modulus with this new assignment
                    added = False
                    for m2 in sorted(missing, key=lambda x: x):
                        if time.time() > deadline:
                            break
                        r2s = find_valid_residues(m2, new_assign)
                        if r2s:
                            new_assign[m2] = r2s[0]
                            new_score = calc_score(new_assign)
                            if new_score > best_score + 1e-12:
                                best_assignment = new_assign
                                best_score = new_score
                                improved = True
                                added = True
                            break
                    
                    if added:
                        break
                
                if improved:
                    break
        
        # Backtracking search for additional improvements
        if time.time() < deadline - 15.0 and iteration % 3 == 0:
            missing = [m for m in all_moduli if m not in best_assignment]
            if missing:
                improved_assign = backtrack_search(best_assignment, missing, max_nodes=10000)
                if improved_assign:
                    new_score = calc_score(improved_assign)
                    if new_score > best_score + 1e-12:
                        best_assignment = improved_assign
                        best_score = new_score
                        improved = True
    
    # Final verification and return
    result = [[m, r] for m, r in best_assignment.items()]
    return result

# EVOLVE-BLOCK-END
