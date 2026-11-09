"""
Improved approach: Greedy + Local Search with Residue Shifting.
The key improvement is allowing the residues of existing moduli to change 
to accommodate new, high-value moduli.
"""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from fractions import Fraction
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Moduli range
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute weights
    weights = {m: 1.0 / m for m in moduli}

    def is_valid(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True

    def find_valid_residue(classes, m):
        """Find any valid residue for m, or None."""
        for r in range(m):
            if is_valid(classes, m, r):
                return r
        return None

    def find_best_residue_slack(classes, m):
        """
        Find residue r for m that maximizes the minimum 'distance' to conflict 
        with existing classes. This helps in packing.
        """
        mi = mod_idx[m]
        best_r = -1
        best_slack = -1
        for r in range(m):
            if not is_valid(classes, m, r):
                continue
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    min_dist = min(min_dist, dist)
            if min_dist > best_slack:
                best_slack = min_dist
                best_r = r
        return best_r

    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        return sum(weights[m] for m, r in classes)

    def greedy_construct(order):
        """Greedy construction: add classes in the given order if possible."""
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            # Try to find a residue that maximizes slack
            r = find_best_residue_slack(classes, m)
            if r is not None and r != -1:
                classes.append([m, r])
                used_moduli.add(m)
        return classes

    def local_search(classes, max_iter=2000):
        """
        Improve by:
        1. Shifting residues to maximize slack (neutral score move, enables future adds).
        2. Adding missing moduli.
        3. Swapping moduli if beneficial.
        4. 'Shift to Add': If a high-value modulus can't be added, try shifting 
           existing residues to make room.
        """
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        # Precompute missing moduli sorted by weight descending
        missing_moduli = [m for m in sorted(moduli, key=weights.get, reverse=True) if m not in used_moduli]
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            new_classes = [list(c) for c in best_classes]
            new_used = set(m for m, r in new_classes)
            
            # --- Move 1: Try to add a missing modulus ---
            # We prioritize adding high-value moduli
            for m_new in missing_moduli[:20]: # Check top 20 missing
                if m_new in new_used:
                    continue
                r_new = find_valid_residue(new_classes, m_new)
                if r_new is not None:
                    # We can add it!
                    new_classes.append([m_new, r_new])
                    new_used.add(m_new)
                    improved = True
                    break
            
            # --- Move 2: If we didn't add, try 'Shift to Add' ---
            if not improved:
                # Try to find a missing modulus that is "almost" addable
                # i.e., conflicts with very few existing moduli
                for m_new in missing_moduli[:10]:
                    if m_new in new_used:
                        continue
                    
                    # Find which existing moduli block all residues of m_new
                    # Actually, simpler: find residues of m_new that are valid 
                    # against most existing classes, and see if we can shift the blockers.
                    
                    # For each residue r_new in [0, m_new):
                    # Count how many existing classes conflict.
                    # If min conflicts is 1 or 2, try to shift those specific classes.
                    
                    best_r_candidate = -1
                    min_conflicts = m_new
                    blockers = []
                    
                    for r_new in range(m_new):
                        confs = []
                        for idx, (m_old, r_old) in enumerate(new_classes):
                            g = gcd_table[mod_idx[m_new]][mod_idx[m_old]]
                            if (r_new - r_old) % g == 0:
                                confs.append(idx)
                        if len(confs) < min_conflicts:
                            min_conflicts = len(confs)
                            best_r_candidate = r_new
                            blockers = confs
                            
                            if min_conflicts == 0:
                                # Found a valid one, this shouldn't happen if Move 1 failed, 
                                # but safety check
                                new_classes.append([m_new, r_new])
                                new_used.add(m_new)
                                improved = True
                                break
                            
                            if min_conflicts <= 2:
                                break # Don't search for 0 if we have 1 or 2, 0 is handled above
                    
                    if improved:
                        break
                    
                    if min_conflicts <= 2 and best_r_candidate != -1:
                        # Try to shift the blockers to free up space for (m_new, best_r_candidate)
                        success = False
                        # We need to find new residues for the blockers such that:
                        # 1. They don't conflict with each other.
                        # 2. They don't conflict with the rest of new_classes.
                        # 3. They don't conflict with (m_new, best_r_candidate).
                        
                        # This is a small sub-CSP. 
                        # Since min_conflicts is small (<=2), we can brute force residues for blockers.
                        
                        other_indices = [i for i in range(len(new_classes)) if i not in blockers]
                        other_classes = [new_classes[i] for i in other_indices]
                        
                        if min_conflicts == 1:
                            idx_b = blockers[0]
                            m_b, r_b_old = new_classes[idx_b]
                            for r_b_new in range(m_b):
                                # Check against m_new
                                g_nm = gcd_table[mod_idx[m_new]][mod_idx[m_b]]
                                if (best_r_candidate - r_b_new) % g_nm == 0:
                                    continue
                                # Check against others
                                valid = True
                                for m_o, r_o in other_classes:
                                    g_bo = gcd_table[mod_idx[m_b]][mod_idx[m_o]]
                                    if (r_b_new - r_o) % g_bo == 0:
                                        valid = False
                                        break
                                if valid:
                                    # Apply shift
                                    new_classes[idx_b][1] = r_b_new
                                    new_classes.append([m_new, best_r_candidate])
                                    new_used.add(m_new)
                                    success = True
                                    break
                        elif min_conflicts == 2:
                            idx_b1, idx_b2 = blockers
                            m_b1, r_b1_old = new_classes[idx_b1]
                            m_b2, r_b2_old = new_classes[idx_b2]
                            
                            # Brute force residues for b1 and b2
                            # To save time, limit search or use heuristics? 
                            # m_b can be up to 200. 200*200 = 40k checks. 
                            # Each check is O(N). N is ~100. 4M ops. Might be too slow if done often.
                            # Let's restrict to a few iterations or smaller moduli first?
                            # Or just try a few random shifts?
                            # Let's try a limited brute force: 100 random pairs or first 50 pairs.
                            
                            found = False
                            # Sort blockers by modulus size, try smaller first?
                            # Actually, just iterate.
                            for r_b1_new in range(m_b1):
                                # Check against m_new
                                g_nm1 = gcd_table[mod_idx[m_new]][mod_idx[m_b1]]
                                if (best_r_candidate - r_b1_new) % g_nm1 == 0:
                                    continue
                                for r_b2_new in range(m_b2):
                                    # Check b1 against b2
                                    g_b1b2 = gcd_table[mod_idx[m_b1]][mod_idx[m_b2]]
                                    if (r_b1_new - r_b2_new) % g_b1b2 == 0:
                                        continue
                                    # Check b2 against m_new
                                    g_nm2 = gcd_table[mod_idx[m_new]][mod_idx[m_b2]]
                                    if (best_r_candidate - r_b2_new) % g_nm2 == 0:
                                        continue
                                    
                                    # Check b1 against others
                                    valid = True
                                    for m_o, r_o in other_classes:
                                        g_bo = gcd_table[mod_idx[m_b1]][mod_idx[m_o]]
                                        if (r_b1_new - r_o) % g_bo == 0:
                                            valid = False
                                            break
                                    if not valid:
                                        continue
                                    
                                    # Check b2 against others
                                    for m_o, r_o in other_classes:
                                        g_bo = gcd_table[mod_idx[m_b2]][mod_idx[m_o]]
                                        if (r_b2_new - r_o) % g_bo == 0:
                                            valid = False
                                            break
                                    if not valid:
                                        continue
                                    
                                    # Success
                                    new_classes[idx_b1][1] = r_b1_new
                                    new_classes[idx_b2][1] = r_b2_new
                                    new_classes.append([m_new, best_r_candidate])
                                    new_used.add(m_new)
                                    found = True
                                    break
                                if found:
                                    break
                            if found:
                                success = True
                        if success:
                            improved = True
                            break
            
            if improved:
                # Update best
                new_score = score(new_classes)
                if new_score > best_score:
                    best_classes = new_classes
                    best_score = new_score
                    used_moduli = new_used
                    # Update missing list
                    missing_moduli = [m for m in sorted(moduli, key=weights.get, reverse=True) if m not in used_moduli]
                    # Occasionally, we might want to re-sort or refresh, but the list is derived from used_moduli
                else:
                    # If score didn't improve (shouldn't happen if we added a modulus), revert
                    new_classes = [list(c) for c in best_classes]
                    new_used = set(m for m, r in new_classes)
            else:
                # No addition possible. Try to optimize residues for slack?
                # Or try swapping?
                # For now, if we can't add, we might be stuck. 
                # Let's try a simple swap: remove a low value, try to add a high value.
                swap_improved = False
                # Sort current classes by weight ascending (worst first)
                sorted_classes = sorted(best_classes, key=lambda x: weights[x[0]])
                for m_old, r_old in sorted_classes[:5]: # Try removing one of the worst 5
                    if m_old in missing_moduli:
                        continue # It's not in best_classes? Wait, sorted_classes is from best_classes.
                    
                    # Temporarily remove m_old
                    temp_classes = [c for c in best_classes if c[0] != m_old]
                    temp_used = set(m for m, r in temp_classes)
                    
                    # Try to add any missing modulus that is better than m_old
                    for m_new in missing_moduli:
                        if weights[m_new] <= weights[m_old]:
                            break # Since missing_moduli is sorted desc, if this one is not better, none are
                    
                    # Actually, missing_moduli contains all missing. We should check if m_new > m_old in weight.
                    # Let's just try to add the best missing one.
                    if not missing_moduli:
                        continue
                    m_new = missing_moduli[0]
                    if weights[m_new] <= weights[m_old] + 1e-9:
                        continue # Not worth swapping
                    
                    # Try to add m_new to temp_classes
                    r_new = find_valid_residue(temp_classes, m_new)
                    if r_new is not None:
                        new_classes = temp_classes + [[m_new, r_new]]
                        new_score = score(new_classes)
                        if new_score > best_score:
                            best_classes = new_classes
                            best_score = new_score
                            used_moduli = set(m for m, r in best_classes)
                            missing_moduli = [m for m in sorted(moduli, key=weights.get, reverse=True) if m not in used_moduli]
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                
                if not swap_improved:
                    # Stuck? Try residue optimization for slack to enable future moves?
                    # This is a neutral move. We'll do it every 10 iterations if stuck.
                    if iteration % 10 == 0:
                        # Try to improve slack of the current solution
                        # This is complex, so we'll skip for now to save time, 
                        # relying on the 'Shift to Add' to handle packing.
                        pass
                    else:
                        # If we've been stuck for a while, break to save time?
                        # Or just break if no improvement in 100 iterations?
                        # Let's just break if no improvement in this specific iteration 
                        # and we've tried swapping.
                        break
        
        return best_classes

    # --- Main Execution ---
    
    # Start with a good greedy solution
    # Order: Descending weight (smallest modulus)
    order = sorted(moduli, key=lambda m: weights[m], reverse=True)
    initial_classes = greedy_construct(order)
    
    best_classes = initial_classes
    best_score = score(best_classes)
    
    # Run local search
    improved_classes = local_search(best_classes, max_iter=1000)
    improved_score = score(improved_classes)
    if improved_score > best_score:
        best_classes = improved_classes
        best_score = improved_score
    
    # Try a few random restarts if time permits
    import random
    random.seed(42)
    
    for restart in range(10):
        if time.time() - start_time > deadline - 10:
            break
        
        # Perturb the initial order or the solution
        # Here we create a random order biased towards small moduli
        remaining = list(moduli)
        order = []
        while remaining:
            # Weighted random choice
            # Smaller moduli have higher weight
            total_w = sum(1.0/m for m in remaining)
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, m in enumerate(remaining):
                cum += 1.0/m
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=500)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = classes
            
    # Final polish
    if time.time() - start_time < deadline - 5:
        final_classes = local_search(best_classes, max_iter=1000)
        final_score = score(final_classes)
        if final_score > best_score:
            best_classes = final_classes
            best_score = final_score

    return best_classes

# EVOLVE-BLOCK-END
