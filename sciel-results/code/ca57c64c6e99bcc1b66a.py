"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    # We want to maximize sum(1/m).
    # Smaller m are more valuable.
    # We process in increasing order of m.
    # This ensures we keep the high-value terms.
    
    selected = []  # List of [m, r]
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    # Greedy: try each modulus in increasing order, pick the smallest residue that works
    # For each m, we try residues 0, 1, 2, ... and pick the first compatible one
    # Since we process small m first, we maximize the sum of 1/m
    
    for m in moduli:
        if time.time() > deadline - 5.0:
            break
        # Try residues in order 0, 1, ..., m-1
        # But we can be smarter: for a given m, different residues give different compatibility
        # We want to pick a residue that is compatible with all already-selected classes
        # and also leaves room for future selections
        # For now, greedy: pick the smallest residue that works
        for r in range(m):
            if is_compatible(m, r, selected):
                selected.append([m, r])
                break
    
    # Calculate score as sum of 1/m
    score = sum(Fraction(1, m) for m, r in selected)
    return selected
# EVOLVE-BLOCK-END
