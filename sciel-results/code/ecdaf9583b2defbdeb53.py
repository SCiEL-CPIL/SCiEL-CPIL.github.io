import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    moduli_list = list(range(20, 201))
    n_moduli = len(moduli_list)
    mod_index = {m: i for i, m in enumerate(moduli_list)}
    
    # Precompute gcd matrix for speed
    gcd_matrix = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            g = math.gcd(moduli_list[i], moduli_list[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
    
    def is_compatible(current, m, r):
        m_idx = mod_index[m]
        for m2, r2 in current:
            m2_idx = mod_index[m2]
            g = gcd_matrix[m_idx][m2_idx]
            if r % g == r2 % g:
                return False
        return True
    
    def find_residue(current, m):
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible(current, m, r):
                return r
        return None
    
    # Strategy 1: Greedy Ascending
    current = []
    score = 0.0
    for m in moduli_list:
        if time.time() > deadline:
            break
        r = find_residue(current, m)
        if r is not None:
            current.append([m, r])
            score += 1.0 / m
    best_score = score
    best_classes = current
    
    # Strategy 2: Randomized greedy with small-modulus bias
    num_restarts = 8000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        strategy = random.random()
        if strategy < 0.35:
            perm = list(moduli_list)
        elif strategy < 0.55:
            perm = list(moduli_list)
            for i in range(n_moduli - 1, 0, -1):
                if random.random() < 0.3:
                    j = random.randint(0, i)
                    perm[i], perm[j] = perm[j], perm[i]
        elif strategy < 0.75:
            perm = list(moduli_list)
            random.shuffle(perm)
        else:
            perm = list(reversed(moduli_list))
        
        current = []
        score = 0.0
        for m in perm:
            if time.time() > deadline:
                break
            r = find_residue(current, m)
            if r is not None:
                current.append([m, r])
                score += 1.0 / m
        
        if score > best_score:
            best_score = score
            best_classes = current
    
    # Strategy 3: Local Search
    if best_classes:
        selected_moduli = set(m for m, r in best_classes)
        unselected_moduli = [m for m in moduli_list if m not in selected_moduli]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for m in unselected_moduli:
                if time.time() > deadline:
                    break
                r = find_residue(best_classes, m)
                if r is not None:
                    best_classes.append([m, r])
                    selected_moduli.add(m)
                    unselected_moduli.remove(m)
                    best_score += 1.0 / m
                    improved = True
            
            if not improved:
                sorted_classes = sorted(best_classes, key=lambda x: x[0], reverse=True)
                for m_remove, r_remove in sorted_classes:
                    if time.time() > deadline:
                        break
                    trial_classes = [c for c in best_classes if c[0] != m_remove]
                    for m_add in unselected_moduli:
                        if m_add >= m_remove:
                            continue
                        r_add = find_residue(trial_classes, m_add)
                        if r_add is not None:
                            new_score = best_score - 1.0 / m_remove + 1.0 / m_add
                            if new_score > best_score:
                                best_classes = trial_classes + [[m_add, r_add]]
                                selected_moduli.remove(m_remove)
                                selected_moduli.add(m_add)
                                unselected_moduli.remove(m_add)
                                unselected_moduli.append(m_remove)
                                best_score = new_score
                                improved = True
                                break
                    if improved:
                        break
    
    return best_classes
