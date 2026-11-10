import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    moduli_range = range(20, 201)
    all_moduli = list(moduli_range)
    
    # Precompute gcds
    gcd_cache = {}
    for i, m1 in enumerate(all_moduli):
        for m2 in all_moduli[i:]:
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)
            gcd_cache[(m2, m1)] = gcd_cache[(m1, m2)]

    def get_gcd(m1, m2):
        return gcd_cache.get((m1, m2), 1)

    def can_add_residue(m_new, r_new, selected):
        for m_sel, r_sel in selected:
            g = get_gcd(m_new, m_sel)
            if r_new % g == r_sel % g:
                return False
        return True

    def find_valid_residue(m_new, selected):
        for r in range(m_new):
            if can_add_residue(m_new, r, selected):
                return r
        return -1

    def is_feasible_set(moduli_set, timeout=0.5):
        """
        Returns list of (m, r) if feasible, else None.
        """
        start = time.time()
        sorted_moduli = sorted(moduli_set)
        
        def backtrack(idx, current_assignment):
            if time.time() - start > timeout:
                return None
            if idx == len(sorted_moduli):
                return current_assignment
            
            m = sorted_moduli[idx]
            for r in range(m):
                valid = True
                for prev_m, prev_r in current_assignment:
                    g = get_gcd(m, prev_m)
                    if r % g == prev_r % g:
                        valid = False
                        break
                if valid:
                    res = backtrack(idx + 1, current_assignment + [(m, r)])
                    if res is not None:
                        return res
            return None

        res = backtrack(0, [])
        if res is not None:
            return res
            
        # Try random orders
        for _ in range(3):
            if time.time() - start > timeout:
                break
            random.shuffle(sorted_moduli)
            res = backtrack(0, [])
            if res is not None:
                return res
        return None

    def score(selected):
        return sum(1.0 / m for m, r in selected)

    best_score = 0.0
    best_selected = []
    
    # Multiple restarts
    # Use system seed for variety
    random.seed()
    
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy construction
        # Sort by 1/m, with slight randomization for ties or near-ties
        sorted_moduli = sorted(all_moduli, key=lambda m: (-1.0/m, random.random()))
        
        current_selected = []
        for m in sorted_moduli:
            r = find_valid_residue(m, current_selected)
            if r != -1:
                current_selected.append((m, r))
                
        current_score = score(current_selected)
        
        # Local Search
        used_moduli = set(m for m, r in current_selected)
        potential_moduli = [m for m in all_moduli if m not in used_moduli]
        potential_moduli.sort(key=lambda m: -1.0/m)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add
            for m_new in potential_moduli:
                if m_new in used_moduli:
                    continue
                r_new = find_valid_residue(m_new, current_selected)
                if r_new != -1:
                    current_selected.append((m_new, r_new))
                    used_moduli.add(m_new)
                    current_score = score(current_selected)
                    improved = True
                    break
            
            # 2. Try to swap
            if not improved:
                # Sort current by m descending (worst first)
                sorted_current = sorted(current_selected, key=lambda x: -x[0])
                for m_old, r_old in sorted_current:
                    # Try to replace with smaller m_new
                    for m_new in potential_moduli:
                        if m_new >= m_old:
                            continue
                        if m_new in used_moduli:
                            continue
                            
                        new_moduli = [m for m, r in current_selected if m != m_old]
                        new_moduli.append(m_new)
                        
                        # Check feasibility
                        feasible_set = is_feasible_set(new_moduli, timeout=0.2)
                        if feasible_set is not None:
                            current_selected = feasible_set
                            used_moduli.discard(m_old)
                            used_moduli.add(m_new)
                            current_score = score(current_selected)
                            improved = True
                            break
                    if improved:
                        break
                        
        if current_score > best_score:
            best_score = current_score
            best_selected = current_selected

    return [[m, r] for m, r in best_selected]
