import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # Safety margin for 90s limit
    
    moduli = list(range(2, 129))
    n_mod = len(moduli)
    
    # Precompute gcd for all modulus pairs
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i != j:
                gcd_table[(m1, m2)] = _gcd(m1, m2)
    
    # Helper to check if a residue r for modulus m is valid against a list of selected (r, m) pairs
    def is_valid(r, m, selected):
        for sr, sm in selected:
            g = gcd_table[(m, sm)]
            if (r - sr) % g == 0:
                return False
        return True

    best_solution = []
    best_count = 0
    
    def try_add_modulus(selected, m):
        """Try to add modulus m to selected by finding a valid residue."""
        for r in range(m):
            if is_valid(r, m, selected):
                selected.append((r, m))
                return True
        return False

    def local_search(selected, time_budget=2.0):
        """
        Improve the solution by:
        1. Trying to add missing moduli.
        2. If stuck, trying to swap residues of existing moduli to enable additions.
        """
        selected = list(selected)
        end_time = time.time() + time_budget
        improved = True
        while improved and time.time() < end_time:
            improved = False
            selected_mods = set(m for _, m in selected)
            
            # Phase 1: Try to add any missing modulus directly
            for m in moduli:
                if m in selected_mods:
                    continue
                if try_add_modulus(selected, m):
                    improved = True
                    break
            
            # Phase 2: If no direct addition, try to swap a residue to enable an addition
            if not improved and time.time() < end_time:
                # Pick a random missing modulus to target
                missing_mods = [m for m in moduli if m not in selected_mods]
                if missing_mods:
                    target_m = random.choice(missing_mods)
                    # Try changing residue of each existing modulus to see if it helps
                    # We pick a random existing modulus to change
                    if selected:
                        idx = random.randint(0, len(selected) - 1)
                        r_old, m_old = selected[idx]
                        
                        # Try all residues for m_old
                        for r_new in range(m_old):
                            if r_new == r_old:
                                continue
                            
                            # Check if r_new is valid against all OTHERS
                            valid_swap = True
                            for jdx in range(len(selected)):
                                if jdx == idx:
                                    continue
                                sr, sm = selected[jdx]
                                g = gcd_table[(m_old, sm)]
                                if (r_new - sr) % g == 0:
                                    valid_swap = False
                                    break
                            
                            if not valid_swap:
                                continue
                            
                            # Perform swap
                            selected[idx] = (r_new, m_old)
                            
                            # Try to add target_m
                            if try_add_modulus(selected, target_m):
                                improved = True
                                break
                            else:
                                # Revert swap if it didn't help
                                selected[idx] = (r_old, m_old)
                        
                        if improved:
                            break
        
        return selected

    # Main search loop
    random.seed(42)
    
    # Strategy 1: Greedy with random ordering
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Random ordering
        perm = moduli[:]
        random.shuffle(perm)
        
        selected = []
        for m in perm:
            try_add_modulus(selected, m)
        
        # Local search
        time_left = deadline - time.time()
        if time_left > 1.0:
            selected = local_search(selected, time_budget=min(2.0, time_left - 0.5))
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = selected[:]
            
            # If we found a very good solution, do more polishing
            if best_count > 50:
                for _ in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    selected = local_search(selected, time_budget=min(1.0, deadline - time.time() - 1.0))
                    if len(selected) > best_count:
                        best_count = len(selected)
                        best_solution = selected[:]

    # Strategy 2: Greedy with "small first" ordering (more constrained)
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        perm = moduli[:]
        # Shuffle within same size? Or just sort by size
        # Let's try sorting by size
        perm.sort()
        
        selected = []
        for m in perm:
            try_add_modulus(selected, m)
        
        time_left = deadline - time.time()
        if time_left > 1.0:
            selected = local_search(selected, time_budget=min(2.0, time_left - 0.5))
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = selected[:]

    if not best_solution:
        # Fallback: just pick 2
        best_solution = [(1, 2)]
    
    return [[r, m] for r, m in best_solution]
