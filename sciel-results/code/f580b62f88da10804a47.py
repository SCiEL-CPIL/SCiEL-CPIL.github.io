"""Rewritten program: Maximum Weight Independent Set via base-modulus packing and local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    moduli = list(range(20, 201))
    
    # Precompute gcd
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            if m1 > m2: continue
            gcd_table[(m1, m2)] = gcd(m1, m2)
            gcd_table[(m2, m1)] = gcd(m1, m2)
            
    def get_gcd(m1, m2):
        return gcd_table[(m1, m2)]

    def can_add(classes, m, r):
        for (m2, r2) in classes:
            g = get_gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total

    def greedy_base(g, max_iter=50):
        """Pack moduli that are multiples of g with pairwise coprime multipliers."""
        # Find all m in [20, 200] divisible by g
        candidates = [m for m in moduli if m % g == 0]
        # Sort by 1/m (descending) to prefer small m
        candidates.sort(key=lambda m: 1/m)
        
        classes = []
        used_k = set()
        
        for m in candidates:
            k = m // g
            # Check if k is coprime to all used_k
            is_coprime = all(gcd(k, uk) == 1 for uk in used_k)
            if not is_coprime:
                continue
            
            # Find a valid residue r
            # r must be distinct mod g from all existing residues
            # Since we are packing base g, and k's are coprime, gcd(m_i, m_j) = g * gcd(k_i, k_j) = g.
            # So condition is r_i != r_j mod g.
            valid_r = None
            for r in range(g):
                if can_add(classes, m, r):
                    valid_r = r
                    break
            
            if valid_r is not None:
                classes.append([m, valid_r])
                used_k.add(k)
                
                # We can have at most g classes for a base g
                if len(classes) >= g:
                    break
                    
        return classes

    # Try all bases
    best_classes = []
    best_score = Fraction(0, 1)
    
    for g in range(2, 21):
        classes = greedy_base(g)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
            
    # Local Search
    def local_search(classes, max_iter=200):
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding a new class
            for m in sorted(moduli, key=lambda x: 1/x):
                if m in used_moduli:
                    continue
                # Try all residues
                for r in range(m):
                    if can_add(best_classes, m, r):
                        new_classes = best_classes + [[m, r]]
                        new_score = score(new_classes)
                        if new_score > best_score:
                            best_classes = new_classes
                            best_score = new_score
                            used_moduli.add(m)
                            improved = True
                            break
                if improved:
                    break
                    
            # Try removing and re-adding
            if not improved and len(best_classes) > 1:
                for i in range(len(best_classes)):
                    if time.time() > deadline: break
                    m_remove = best_classes[i][0]
                    temp_classes = [c for j, c in enumerate(best_classes) if j != i]
                    temp_used = set(m for m, r in temp_classes)
                    
                    for m_add in sorted(moduli, key=lambda x: 1/x):
                        if m_add in temp_used:
                            continue
                        for r_add in range(m_add):
                            if can_add(temp_classes, m_add, r_add):
                                new_classes = temp_classes + [[m_add, r_add]]
                                new_score = score(new_classes)
                                if new_score > best_score:
                                    best_classes = new_classes
                                    best_score = new_score
                                    used_moduli = set(m for m, r in best_classes)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                        
            if not improved:
                break
        return best_classes

    best_classes = local_search(best_classes, max_iter=300)
    best_score = score(best_classes)
    
    return best_classes
# EVOLVE-BLOCK-END
