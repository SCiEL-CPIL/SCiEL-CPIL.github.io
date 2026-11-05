import math
import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    def is_valid_assignment(selected, m, r):
        """Check if adding (m, r) to selected keeps all pairs disjoint."""
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def find_valid_residue(selected, m):
        """Find a valid residue for modulus m given current selected set."""
        for r in range(m):
            if is_valid_assignment(selected, m, r):
                return r
        return None
    
    def compute_score(selected):
        return sum(1.0 / m for m, r in selected)
    
    # Phase 1: Greedy construction
    selected = []
    for m in range(20, 201):
        r = find_valid_residue(selected, m)
        if r is not None:
            selected.append((m, r))
    
    best_selected = list(selected)
    best_score = compute_score(best_selected)
    
    # Phase 2: Local search with insert and swap moves
    # Try multiple random restarts with different orderings
    random.seed(42)
    
    def local_search(initial_selected):
        current = list(initial_selected)
        current_score = compute_score(current)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline - 10:
            improved = False
            no_improve_count = 0
            
            # Try insert moves: add a missing modulus
            selected_moduli = {m for m, r in current}
            for m in range(20, 201):
                if m in selected_moduli:
                    continue
                r = find_valid_residue(current, m)
                if r is not None:
                    new_score = current_score + 1.0 / m
                    if new_score > current_score + 1e-12:
                        current.append((m, r))
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
            
            if improved:
                continue
            
            # Try swap moves: change residue of an existing modulus
            for i in range(len(current)):
                m_i, r_i = current[i]
                # Try different residues for m_i
                for r_new in range(m_i):
                    if r_new == r_i:
                        continue
                    # Check if swapping r_i to r_new is valid
                    valid = True
                    for j in range(len(current)):
                        if i == j:
                            continue
                        m_j, r_j = current[j]
                        g = math.gcd(m_i, m_j)
                        if r_new % g == r_j % g:
                            valid = False
                            break
                    if valid:
                        # Score doesn't change with residue, but may enable future inserts
                        # Only accept if it enables an insert
                        current[i] = (m_i, r_new)
                        # Try to insert a missing modulus
                        selected_moduli = {m for m, r in current}
                        for m in range(20, 201):
                            if m in selected_moduli:
                                continue
                            r = find_valid_residue(current, m)
                            if r is not None:
                                current.append((m, r))
                                current_score += 1.0 / m
                                improved = True
                                no_improve_count = 0
                                break
                        if not improved:
                            current[i] = (m_i, r_i)
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
        
        return current, current_score
    
    # Run local search from greedy solution
    result, score = local_search(best_selected)
    if score > best_score:
        best_selected = result
        best_score = score
    
    # Try a few random orderings for greedy + local search
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        
        # Random permutation of moduli
        moduli = list(range(20, 201))
        random.shuffle(moduli)
        
        greedy_sel = []
        for m in moduli:
            r = find_valid_residue(greedy_sel, m)
            if r is not None:
                greedy_sel.append((m, r))
        
        result, score = local_search(greedy_sel)
        if score > best_score:
            best_selected = result
            best_score = score
    
    return [[m, r] for m, r in best_selected]
