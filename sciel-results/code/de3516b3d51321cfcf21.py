import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    best_selected = []
    best_score = 0.0

    def is_compatible(m_new, r_new, selected_list):
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def run_random_greedy():
        selected = []
        # We process in increasing order of m to prioritize high value terms
        for m in moduli:
            # Find all valid residues
            valid_residues = []
            for r in range(m):
                if is_compatible(m, r, selected):
                    valid_residues.append(r)
            
            if valid_residues:
                # Pick a random valid residue
                r = random.choice(valid_residues)
                selected.append([m, r])
            # If no valid residue, we skip this modulus.
            # This is a limitation: we don't backtrack.
        
        score = sum(Fraction(1, m) for m, r in selected)
        return selected, float(score)

    # Run multiple restarts
    # Estimate time per restart:
    # 181 moduli, max 200 residues each.
    # is_compatible checks up to 181 items.
    # 181 * 200 * 181 ~ 6.5 million operations per restart.
    # In Python, this might take ~0.05s - 0.1s per restart.
    # 110s / 0.1s = 1100 restarts.
    
    restarts = 0
    while time.time() < deadline - 5.0:
        selected, score = run_random_greedy()
        if score > best_score:
            best_score = score
            best_selected = selected
        restarts += 1
        
    # If we ran very few restarts, maybe we should try a deterministic greedy first?
    # The deterministic greedy is just one specific random choice (e.g., always pick smallest).
    # We can add a deterministic run to ensure we have a valid baseline.
    if restarts == 0:
        selected, score = run_random_greedy()
        best_selected = selected
        best_score = score

    return best_selected
