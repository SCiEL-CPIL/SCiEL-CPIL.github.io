"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd_table[i][j] = gcd(moduli[i], moduli[j])
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_valid_residue(classes, m):
        """Find a valid residue for modulus m given existing classes.
        Returns the residue that minimizes conflicts (maximizes future flexibility)."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None
        
        # Choose residue that avoids most "tight" constraints
        # A residue is "tight" if it's close to being congruent to another class mod gcd
        # We want to maximize minimum distance to any forbidden residue class
        best_r = valid_residues[0]
        best_score = -1
        
        for r in valid_residues:
            min_dist = m  # maximum possible distance
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                # The forbidden residues are those where (r - r2) % g == 0
                # i.e., r ≡ r2 (mod g)
                # Distance from r to nearest forbidden residue in [0, m)
                # Forbidden residues: r2 mod g, r2+g mod g, ... 
                # Actually all residues r' where r' ≡ r2 (mod g)
                # The "distance" in terms of how many valid choices remain
                # is related to g. Larger g means more constraints.
                # We want residues that are "far" from all r2 mod g in some sense.
                # Simpler: just pick the first valid one, or use a heuristic.
                pass
            
            # Heuristic: prefer residues that are not 0 (0 is often more constrained)
            # and prefer smaller residues for determinism
            s = 1000 - abs(r - m//2)  # prefer middle-ish residues
            if s > best_score:
                best_score = s
                best_r = r
        
        return best_r
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli, key=lambda m: 1/m)  # smallest moduli first
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r = find_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes
    
    def local_search(classes, max_iter=500):
        """Try to improve by swapping residues or removing/adding classes."""
        best_score = score(classes)
        best_classes = [list(c) for c in classes]
        used_moduli = set(m for m, r in best_classes)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new class (smallest moduli first for better score)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r = find_valid_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = score(new_classes)
                    if new_score > best_score:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # Try 2-opt: remove one class, try to add two better ones
            if not improved and len(best_classes) > 2:
                for i in range(len(best_classes)):
                    if time.time() > deadline:
                        break
                    m_remove, r_remove = best_classes[i]
                    # Temporarily remove
                    temp_classes = [c for j, c in enumerate(best_classes) if j != i]
                    temp_used = set(m for m, r in temp_classes)
                    
                    # Try to add two classes that together are better than m_remove
                    added_any = False
                    for m1 in sorted(moduli):
                        if m1 in temp_used or m1 == m_remove:
                            continue
                        r1 = find_valid_residue(temp_classes, m1)
                        if r1 is not None:
                            temp_classes2 = temp_classes + [[m1, r1]]
                            temp_used2 = temp_used | {m1}
                            
                            for m2 in sorted(moduli):
                                if m2 in temp_used2 or m2 == m_remove:
                                    continue
                                r2 = find_valid_residue(temp_classes2, m2)
                                if r2 is not None:
                                    new_classes = temp_classes2 + [[m2, r2]]
                                    new_score = score(new_classes)
                                    if new_score > best_score:
                                        best_classes = new_classes
                                        best_score = new_score
                                        used_moduli = set(m for m, r in best_classes)
                                        improved = True
                                        added_any = True
                                        break
                        if added_any:
                            break
                    if added_any:
                        break
            
            if not improved:
                break
        
        return best_classes
    
    # Multi-start with different orderings
    best_classes = None
    best_score = Fraction(0, 1)
    
    # Start with greedy (smallest moduli first)
    classes = greedy_construct()
    classes = local_search(classes, max_iter=200)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Try randomized greedy multiple times
    random.seed(42)
    num_starts = 100
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        # Create a random ordering with bias toward small moduli
        # Use exponential weighting to strongly favor small moduli
        weights = [1.0 / (m ** 1.5) for m in moduli]
        order = []
        remaining = list(moduli)
        remaining_weights = list(weights)
        while remaining:
            total_w = sum(remaining_weights)
            if total_w <= 0:
                order.append(remaining[0])
                remaining.pop(0)
                remaining_weights.pop(0)
                continue
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
            remaining_weights.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=100)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final local search on best solution
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=500)
        best_score = score(best_classes)
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
