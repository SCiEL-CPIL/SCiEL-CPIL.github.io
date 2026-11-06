import math
import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    moduli = list(range(2, 129))
    n = len(moduli)
    
    # Precompute GCDs
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = _gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    best_count = 0
    best_result = []
    
    def is_valid(i, r, selected):
        """Check if residue r for modulus index i is disjoint from all in selected."""
        for (j, rj) in selected:
            g = gcds[i][j]
            if (r - rj) % g == 0:
                return False
        return True
    
    def greedy_solve(order):
        """Greedy: process moduli in given order, pick first valid residue."""
        selected = []
        for i in order:
            m = moduli[i]
            for r in range(m):
                if is_valid(i, r, selected):
                    selected.append((i, r))
                    break
        return selected
    
    def local_search(selected, time_budget=5.0):
        """Improve by trying to add missing moduli with different residues, or swapping."""
        selected = list(selected)
        end_time = time.time() + time_budget
        improved = True
        while improved and time.time() < end_time:
            improved = False
            in_set = set(i for i, _ in selected)
            
            # Try to add a missing modulus
            for i in range(n):
                if i in in_set:
                    continue
                m = moduli[i]
                # Try all residues for this modulus
                for r in range(m):
                    if is_valid(i, r, selected):
                        selected.append((i, r))
                        improved = True
                        break
                if improved:
                    break
            
            # Try to swap residues to enable adding more
            if not improved and time.time() < end_time:
                for idx, (i, r) in enumerate(selected):
                    m = moduli[i]
                    found_swap = False
                    for r_new in range(m):
                        if r_new == r:
                            continue
                        # Check if r_new is valid against all others
                        valid = True
                        for j, (k, rk) in enumerate(selected):
                            if j == idx:
                                continue
                            g = gcds[i][k]
                            if (r_new - rk) % g == 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Try swapping and see if we can add a new modulus
                        old_r = selected[idx][1]
                        selected[idx] = (i, r_new)
                        
                        # Try to add a missing modulus
                        in_set_after_swap = set(i for i, _ in selected)
                        for i2 in range(n):
                            if i2 in in_set_after_swap:
                                continue
                            m2 = moduli[i2]
                            for r2 in range(m2):
                                if is_valid(i2, r2, selected):
                                    selected.append((i2, r2))
                                    found_swap = True
                                    break
                            if found_swap:
                                break
                        
                        if not found_swap:
                            # Revert the swap
                            selected[idx] = (i, old_r)
                    
                    if found_swap:
                        improved = True
                        break
        
        return selected
    
    def most_constrained_order():
        """Order moduli by how few residues they have (most constrained first)."""
        # Count how many residues are blocked for each modulus by the current set
        # For now, use a heuristic: smaller moduli have fewer residues to choose from
        indices = list(range(n))
        # Sort by modulus size (smaller first) as a proxy for constraint
        indices.sort(key=lambda i: moduli[i])
        return indices
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Strategy 1: Random order
        if attempts % 3 == 1:
            indices = list(range(n))
            random.shuffle(indices)
            selected = greedy_solve(indices)
        # Strategy 2: Most constrained first
        elif attempts % 3 == 2:
            selected = greedy_solve(most_constrained_order())
        # Strategy 3: Reverse order (largest first)
        else:
            indices = list(range(n))
            random.shuffle(indices)
            indices.sort(key=lambda i: moduli[i], reverse=True)
            selected = greedy_solve(indices)
        
        # Polish with local search
        if len(selected) > best_count or (best_count > 0 and len(selected) >= best_count):
            selected_polished = local_search(selected, time_budget=min(3.0, max(0.5, deadline - time.time() - 2.0)))
            if len(selected_polished) > len(selected):
                selected = selected_polished
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [[r, moduli[i]] for i, r in selected]
            
            # If we've found a very large set, try to polish more
            if best_count > 100:
                for _ in range(3):
                    if time.time() > deadline - 3:
                        break
                    indices = list(range(n))
                    random.shuffle(indices)
                    selected = greedy_solve(indices)
                    selected_polished = local_search(selected, time_budget=min(2.0, max(0.5, deadline - time.time() - 2.0)))
                    if len(selected_polished) > best_count:
                        best_count = len(selected_polished)
                        best_result = [[r, moduli[i]] for i, r in selected_polished]
                break
    
    if not best_result:
        # Fallback
        return [[1, 2]]
        
    return best_result
