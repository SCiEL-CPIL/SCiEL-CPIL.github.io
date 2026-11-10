import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    candidates = list(range(20, 201))
    n = len(candidates)

    # Precompute gcd matrix
    gcd_matrix = [[gcd(candidates[i], candidates[j]) for j in range(n)] for i in range(n)]

    import random

    def is_valid_r(m_idx, r, selected, selected_indices):
        """Check if residue r for modulus at m_idx is valid with all selected."""
        m = candidates[m_idx]
        for sel_idx in selected_indices:
            m2 = candidates[sel_idx]
            r2 = selected[sel_idx]
            g = gcd_matrix[m_idx][sel_idx]
            if r % g == r2 % g:
                return False
        return True

    def get_valid_rs(m_idx, selected, selected_indices):
        """Get all valid residues for modulus at m_idx."""
        m = candidates[m_idx]
        valid_rs = []
        for r in range(m):
            if is_valid_r(m_idx, r, selected, selected_indices):
                valid_rs.append(r)
        return valid_rs

    def greedy_solve(order, seed):
        """Greedy: process moduli in given order, pick random valid residue."""
        rng = random.Random(seed)
        selected = [None] * n
        selected_indices = []
        
        for m_idx in order:
            valid_rs = get_valid_rs(m_idx, selected, selected_indices)
            if not valid_rs:
                continue
            r = rng.choice(valid_rs)
            selected[m_idx] = r
            selected_indices.append(m_idx)
        
        # Build result
        result = []
        for m_idx in selected_indices:
            m = candidates[m_idx]
            r = selected[m_idx]
            result.append([m, r])
        return result, selected, selected_indices

    def local_search(selected, selected_indices, max_iters=500, seed=0):
        """Try to add unselected moduli by adjusting residues."""
        rng = random.Random(seed)
        score = sum(1.0 / candidates[i] for i in selected_indices)
        best_score = score
        best_selected = selected[:]
        best_indices = selected_indices[:]
        
        # Unselected moduli sorted by decreasing 1/m (most valuable first)
        unselected = [i for i in range(n) if i not in set(selected_indices)]
        unselected.sort(key=lambda i: -1.0 / candidates[i])
        
        for iter_num in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add an unselected modulus
            for target_idx in unselected[:20]:  # Try top 20 most valuable unselected
                m = candidates[target_idx]
                # First check if there's a valid residue without changing anything
                valid_rs = get_valid_rs(target_idx, selected, selected_indices)
                if valid_rs:
                    r = rng.choice(valid_rs)
                    selected[target_idx] = r
                    selected_indices.append(target_idx)
                    unselected.remove(target_idx)
                    score += 1.0 / m
                    if score > best_score:
                        best_score = score
                        best_selected = selected[:]
                        best_indices = selected_indices[:]
                    improved = True
                    continue
                
                # No valid residue exists; try to adjust one existing residue to free space
                # Try adjusting each selected modulus's residue
                for sel_idx in selected_indices:
                    m2 = candidates[sel_idx]
                    g = gcd_matrix[target_idx][sel_idx]
                    # Current residue r2 = selected[sel_idx]
                    r2 = selected[sel_idx]
                    # We need r2 % g != target_r % g for all valid target_r
                    # Try changing r2 to a new value r2_new such that:
                    # 1. r2_new is valid with all other selected moduli
                    # 2. There exists some r_target valid with all selected (with r2_new)
                    
                    # Try a few random new residues for sel_idx
                    for _ in range(3):
                        r2_new = rng.randrange(m2)
                        if r2_new == r2:
                            continue
                        
                        # Check if r2_new is valid with all other selected
                        valid = True
                        for other_idx in selected_indices:
                            if other_idx == sel_idx:
                                continue
                            g_other = gcd_matrix[sel_idx][other_idx]
                            if r2_new % g_other == selected[other_idx] % g_other:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Now check if there's a valid residue for target_idx with r2_new
                        temp_selected = selected[:]
                        temp_selected[sel_idx] = r2_new
                        valid_target_rs = get_valid_rs(target_idx, temp_selected, selected_indices)
                        if valid_target_rs:
                            # Apply the change
                            selected[sel_idx] = r2_new
                            r_target = rng.choice(valid_target_rs)
                            selected[target_idx] = r_target
                            selected_indices.append(target_idx)
                            unselected.remove(target_idx)
                            score += 1.0 / m
                            if score > best_score:
                                best_score = score
                                best_selected = selected[:]
                                best_indices = selected_indices[:]
                            improved = True
                            break
                    if improved:
                        break
                
                if time.time() > deadline:
                    break
            
            if not improved:
                # No improvement this iteration; try a random perturbation
                if selected_indices:
                    # Randomly change one residue
                    idx = rng.choice(selected_indices)
                    m = candidates[idx]
                    new_r = rng.randrange(m)
                    # Check validity
                    if is_valid_r(idx, new_r, selected, [i for i in selected_indices if i != idx]):
                        selected[idx] = new_r
                    else:
                        # Revert and try another
                        pass
        
        return best_selected, best_indices, best_score

    best_score = 0.0
    best_solution = []
    
    # Multiple restarts with different strategies
    for restart in range(80):
        if time.time() > deadline:
            break
        
        seed = restart * 12345 + 67890
        
        if restart < 40:
            # Standard increasing order
            order = list(range(n))
        elif restart < 60:
            # Randomized order biased toward small moduli
            order = list(range(n))
            rng = random.Random(seed)
            for i in range(n):
                # Swap with a random position, biased toward later positions for small i
                if i < n // 2:
                    j = rng.randint(i, min(i + 10, n - 1))
                    order[i], order[j] = order[j], order[i]
            order = sorted(order, key=lambda x: candidates[x] if rng.random() < 0.7 else rng.random() * 200)
        else:
            # Fully random order
            order = list(range(n))
            random.Random(seed).shuffle(order)
        
        result, selected, selected_indices = greedy_solve(order, seed)
        score = sum(1.0 / candidates[i] for i in selected_indices)
        
        if score > best_score:
            best_score = score
            best_solution = result
        
        # Local search to improve
        if len(selected_indices) < n:  # Only if not all moduli are selected
            improved_selected, improved_indices, improved_score = local_search(
                selected, selected_indices, max_iters=100, seed=seed
            )
            if improved_score > best_score:
                best_score = improved_score
                best_result = []
                for i in improved_indices:
                    best_result.append([candidates[i], improved_selected[i]])
                best_solution = best_result
    
    if not best_solution:
        return [[20, 0]]
    
    return best_solution
