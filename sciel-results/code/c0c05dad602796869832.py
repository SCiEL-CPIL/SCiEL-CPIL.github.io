"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
from math import gcd
import time

def construct():
    """
    Find a maximum-weight set of pairwise disjoint residue classes with distinct moduli in [20, 200].
    Weight of class (m, r) is 1/m.
    
    Approach:
    1. Greedy construction: process moduli in order of decreasing weight (increasing m),
       for each modulus try all residues and pick the first compatible one.
    2. Local search: try to swap residues or replace classes to improve the total weight.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    all_moduli = list(range(20, 201))
    
    def is_compatible(m1, r1, m2, r2):
        """Check if residue classes (m1, r1) and (m2, r2) are disjoint."""
        g = gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def is_compatible_with_all(m, r, selected):
        """Check if (m, r) is compatible with all classes in selected."""
        for m2, r2 in selected:
            if not is_compatible(m, r, m2, r2):
                return False
        return True
    
    # Greedy construction: try to include all moduli
    # Process in order of increasing m (highest weight 1/m first)
    selected = []  # list of [m, r]
    
    for m in sorted(all_moduli, key=lambda x: -1.0/x):  # decreasing weight = increasing m... wait
        # Actually, 1/m is larger for smaller m, so we want smaller m first
        pass
    
    # Re-do: smaller m has larger 1/m, so process smaller m first for greedy
    for m in sorted(all_moduli, key=lambda x: x):  # increasing m = decreasing 1/m
        if time.time() > deadline:
            break
        found = False
        for r in range(m):
            if is_compatible_with_all(m, r, selected):
                selected.append([m, r])
                found = True
                break
    
    # Now try to add more moduli that weren't included
    selected_moduli = set(m for m, r in selected)
    remaining_moduli = [m for m in all_moduli if m not in selected_moduli]
    
    for m in remaining_moduli:
        if time.time() > deadline:
            break
        for r in range(m):
            if is_compatible_with_all(m, r, selected):
                selected.append([m, r])
                break
    
    # Local search: try to improve by swapping residues
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            mi, ri = selected[i]
            # Try changing the residue of class i
            for r_new in range(mi):
                if r_new == ri:
                    continue
                # Check if new residue is compatible with all other classes
                compatible = True
                for j in range(len(selected)):
                    if j == i:
                        continue
                    mj, rj = selected[j]
                    if not is_compatible(mi, r_new, mj, rj):
                        compatible = False
                        break
                if compatible:
                    # This doesn't change the score, but might enable further improvements
                    # Actually, changing residue doesn't change 1/m, so skip for now
                    pass
        
        # Try to replace a class with a different modulus
        for i in range(len(selected)):
            mi, ri = selected[i]
            for m_new in all_moduli:
                if m_new == mi:
                    continue
                if m_new in selected_moduli:
                    continue
                # Check if we can find a residue for m_new that's compatible with all others
                # except we're replacing mi
                for r_new in range(m_new):
                    compatible = True
                    for j in range(len(selected)):
                        if j == i:
                            continue
                        m_j, r_j = selected[j]
                        # Check compatibility using CRT
                        if (r_new - r_j) % math.gcd(m_new, m_j) != 0:
                            compatible = False
                            break
                    if compatible:
                        # Add this modulus and residue
                        selected[i] = (m_new, r_new)
                        selected_moduli.add(m_new)
                        selected_moduli.discard(mi)
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    # Simpler local search: try to add any remaining modulus
    selected_moduli = set(m for m, r in selected)
    for m in all_moduli:
        if m in selected_moduli:
            continue
        for r in range(m):
            if is_compatible_with_all(m, r, selected):
                selected.append([m, r])
                selected_moduli.add(m)
                break
    
    return selected
# EVOLVE-BLOCK-END
