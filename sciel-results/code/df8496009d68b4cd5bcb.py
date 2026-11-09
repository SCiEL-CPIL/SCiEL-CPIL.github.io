"""Model-and-optimize approach for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd
from fractions import Fraction

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
    
    def find_best_residue_fast(classes, m):
        """
        Find a valid residue for modulus m given existing classes.
        Returns (residue, slack) or (None, 0).
        Uses boolean masking for speed.
        """
        if not classes:
            return 0, m
        
        mi = mod_idx[m]
        valid = [True] * m
        
        # Mark invalid residues
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if g == 1:
                continue # Always compatible
            bad_res = r2 % g
            # Mark all residues congruent to bad_res mod g as invalid
            for k in range(bad_res, m, g):
                valid[k] = False
        
        # Find valid residues
        valid_residues = [r for r in range(m) if valid[r]]
        
        if not valid_residues:
            return None, 0
        
        if len(valid_residues) == 1:
            return valid_residues[0], 0
        
        # Score by slack: min distance to any forbidden residue mod g
        best_r = valid_residues[0]
        best_slack = -1
        
        # Precompute forbidden residues for each class to speed up slack calc
        # forbidden[i] = set of r % g_i that are bad
        # Actually, we can just iterate.
        for r in valid_residues:
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    if dist < min_dist:
                        min_dist = dist
            if min_dist > best_slack:
                best_slack = min_dist
                best_r = r
        
        return best_r, best_slack

    def score(classes):
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total

    def greedy_construct(order=None):
        if order is None:
            # Sort by increasing modulus to prioritize high 1/m
            order = sorted(moduli)
        
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r, slack = find_best_residue_fast(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes

    def try_add_all(classes):
        used_moduli = set(m for m, r in classes)
        remaining = [m for m in sorted(moduli) if m not in used_moduli]
        
        for m in remaining:
            if time.time() - start_time > deadline - 2:
                break
            r, slack = find_best_residue_fast(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        
        return classes

    def local_search(classes, max_iter=1000):
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            n_classes = len(best_classes)
            
            # 1. Try adding a new modulus (smallest first)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r, slack = find_best_residue_fast(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0/m
                    if new_score > best_score + 1e-12:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # 2. Try 1-for-1 swap
            if not improved:
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                for m_new in missing:
                    if time.time() - start_time > deadline - 3:
                        break
                    # Try removing each existing class to see if m_new fits
                    # We prioritize removing large moduli (low 1/m)
                    existing_sorted = sorted(best_classes, key=lambda c: c[0], reverse=True)
                    for i, (m_old, r_old) in enumerate(existing_sorted):
                        if 1.0/m_new <= 1.0/m_old + 1e-12:
                            continue # No gain
                        temp_classes = [list(c) for j, c in enumerate(best_classes) if j != i]
                        r_new, _ = find_best_residue_fast(temp_classes, m_new)
                        if r_new is not None:
                            new_classes = temp_classes + [[m_new, r_new]]
                            new_score = score(new_classes)
                            if new_score > best_score + 1e-12:
                                best_classes = new_classes
                                best_score = new_score
                                used_moduli = set(m for m, r in best_classes)
                                improved = True
                                break
                    if improved:
                        break
            
            # 3. Residue re-optimization
            if not improved:
                for i in range(n_classes):
                    m, r = best_classes[i]
                    other_classes = [c for j, c in enumerate(best_classes) if j != i]
                    new_r, _ = find_best_residue_fast(other_classes, m)
                    if new_r is not None and new_r != r:
                        # Accept if it enables an addition
                        temp_classes = [list(c) for j, c in enumerate(best_classes) if j != i] + [[m, new_r]]
                        used_temp = set(m2 for m2, r2 in temp_classes)
                        for m_add in sorted(moduli):
                            if m_add in used_temp:
                                continue
                            r_add, _ = find_best_residue_fast(temp_classes, m_add)
                            if r_add is not None:
                                best_classes = temp_classes + [[m_add, r_add]]
                                best_score = score(best_classes)
                                used_moduli = set(m2 for m2, r2 in best_classes)
                                improved = True
                                break
                        if improved:
                            break
                        else:
                            # Still update residue if it improves slack (neutral move)
                            best_classes[i] = [m, new_r]
            
            if not improved:
                break
        
        return best_classes

    # Multi-start
    best_classes = None
    best_score = 0.0
    
    # Start 1: Greedy smallest first
    classes = greedy_construct()
    classes = try_add_all(classes)
    classes = local_search(classes, max_iter=2000)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Random starts
    random.seed(42)
    num_starts = 50
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 25:
            break
        
        # Random ordering biased toward small moduli
        remaining = list(moduli)
        order = []
        while remaining:
            alpha = 1.0 + start_idx * 0.1
            weights = [1.0 / (m ** alpha) for m in remaining]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = try_add_all(classes)
        
        if time.time() - start_time > deadline - 15:
            s = score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
            continue
        
        classes = local_search(classes, max_iter=1000)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final polish
    if best_classes is not None:
        for round_num in range(5):
            if time.time() - start_time > deadline - 2:
                break
            improved_classes = local_search(best_classes, max_iter=1500)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break

    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
