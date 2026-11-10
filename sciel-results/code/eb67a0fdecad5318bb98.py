import math
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill

    # Precompute gcd for all pairs of moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i <= j:
                gcd_table[(i, j)] = math.gcd(m1, m2)
                gcd_table[(j, i)] = gcd_table[(i, j)]

    # Helper: check if (m, r) is compatible with a list of (m, r)
    def is_compatible(m, r, sel):
        for m2, r2 in sel:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def calc_score(sel):
        return sum(1.0 / m for m, r in sel)

    # Greedy initialization: try to add moduli in ascending order
    def greedy_init():
        selected = []
        for m in moduli:
            found = False
            for r in range(m):
                if is_compatible(m, r, selected):
                    selected.append((m, r))
                    found = True
                    break
        return selected

    # Local search with additions, removals, and residue changes
    def local_search(initial, time_budget):
        current = list(initial)
        used_moduli = set(m for m, r in current)
        current_score = calc_score(current)
        best_score = current_score
        best_selected = list(current)
        
        # Track which moduli are used
        attempts = 0
        max_attempts = 500
        
        while time.time() - start_time < time_budget:
            improved = False
            
            # Try to add a random unselected modulus
            for _ in range(max_attempts):
                if time.time() - start_time > time_budget:
                    break
                    
                m = random.choice(moduli)
                if m in used_moduli:
                    continue
                
                # Try a random residue
                r = random.randint(0, m - 1)
                
                if is_compatible(m, r, current):
                    current.append((m, r))
                    used_moduli.add(m)
                    current_score += 1.0 / m
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(current)
                    break
            
            if improved:
                continue
            
            # No addition found; perform a kick
            if len(current) > 0:
                kick_type = random.random()
                if kick_type < 0.3:
                    # Remove a random element
                    idx = random.randint(0, len(current) - 1)
                    removed_m, removed_r = current.pop(idx)
                    used_moduli.remove(removed_m)
                    current_score -= 1.0 / removed_m
                elif kick_type < 0.6:
                    # Change residue of a random element
                    idx = random.randint(0, len(current) - 1)
                    m, r_old = current[idx]
                    r_new = random.randint(0, m - 1)
                    if r_new != r_old:
                        # Check compatibility with rest
                        compatible = True
                        for i, (m2, r2) in enumerate(current):
                            if i == idx:
                                continue
                            g = math.gcd(m, m2)
                            if r_new % g == r2 % g:
                                compatible = False
                                break
                        if compatible:
                            current[idx] = (m, r_new)
                        else:
                            # Revert (no change)
                            pass
                else:
                    # Swap: remove one, try to add another
                    idx = random.randint(0, len(current) - 1)
                    removed_m, removed_r = current.pop(idx)
                    used_moduli.remove(removed_m)
                    current_score -= 1.0 / removed_m
                    
                    # Try to add a different modulus
                    for _ in range(20):
                        m_new = random.choice(moduli)
                        if m_new in used_moduli:
                            continue
                        r_new = random.randint(0, m_new - 1)
                        if is_compatible(m_new, r_new, current):
                            current.append((m_new, r_new))
                            used_moduli.add(m_new)
                            current_score += 1.0 / m_new
                            if current_score > best_score:
                                best_score = current_score
                                best_selected = list(current)
                            break
                    else:
                        # Could not add; put back
                        current.append((removed_m, removed_r))
                        used_moduli.add(removed_m)
                        current_score += 1.0 / removed_m
            else:
                break
        
        return best_selected, best_score

    # Multi-restart strategy
    best_overall = greedy_init()
    best_score_overall = calc_score(best_overall)
    
    # Run local search on the greedy result
    time_per_restart = 20.0
    num_restarts = int((time_limit - 5.0) / time_per_restart)
    
    for restart in range(num_restarts):
        time_budget = start_time + time_limit - 5.0
        if time.time() > time_budget:
            break
        
        if restart == 0:
            initial = best_overall
        else:
            # Create a perturbed initial solution
            initial = list(best_overall)
            # Randomly remove some elements
            num_remove = random.randint(0, min(5, len(initial)))
            for _ in range(num_remove):
                if initial:
                    idx = random.randint(0, len(initial) - 1)
                    initial.pop(idx)
        
        result, score = local_search(initial, time_budget)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = result
    
    return [[m, r] for m, r in best_overall]
