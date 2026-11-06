"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_compatible(x, selected_set):
    for y in selected_set:
        if _lcm(x, y) <= 256:
            return False
    return True

def _reciprocal_sum(selected):
    return sum(1.0 / x for x in selected)

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Universe: [2, 256]
    universe = list(range(2, 257))
    
    # Precompute lcm matrix for speed
    n = 255  # indices 0..254 correspond to values 2..256
    # lcm_val[i][j] = lcm(universe[i], universe[j])
    # Use a 2D list but only store >256 as boolean for compatibility
    # compat[i][j] = True if lcm(universe[i], universe[j]) > 256
    compat = [[True] * n for _ in range(n)]
    for i in range(n):
        ui = universe[i]
        for j in range(i + 1, n):
            uj = universe[j]
            if _lcm(ui, uj) <= 256:
                compat[i][j] = False
                compat[j][i] = False
    
    def _check_all_compatible(selected_indices):
        """Check if all pairs in selected_indices are compatible."""
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                if not compat[selected_indices[i]][selected_indices[j]]:
                    return False
        return True
    
    def _swap_improvement(selected_indices, remove_idx, add_val):
        """Check if removing selected_indices[remove_idx] and adding add_val improves sum."""
        old_val = universe[selected_indices[remove_idx]]
        if add_val == old_val:
            return False
        # Check compatibility of add_val with all remaining
        for i in range(len(selected_indices)):
            if i == remove_idx:
                continue
            # Find index of add_val in universe
            # add_val is a value, need its index
            add_idx = add_val - 2  # since universe[i] = i+2
            if not compat[add_idx][selected_indices[i]]:
                return False
        # Check improvement
        delta = 1.0 / add_val - 1.0 / old_val
        return delta > 0
    
    # Greedy initialization: prioritize by 1/x (small x first)
    # Try multiple greedy orders
    best_selected = []
    best_score = 0.0
    
    # Order 1: ascending (small numbers first)
    order1 = list(range(n))  # universe[0]=2, universe[1]=3, ...
    # Order 2: descending reciprocal = ascending value (same as order1 for this purpose)
    # Order 3: try a different heuristic - numbers that are "less constraining"
    
    for order in [order1]:
        selected = []
        selected_set = set()
        for idx in order:
            val = universe[idx]
            if _is_compatible(val, selected_set):
                selected.append(idx)
                selected_set.add(val)
        
        score = sum(1.0 / universe[i] for i in selected)
        if score > best_score:
            best_score = score
            best_selected = list(selected)
    
    # Local search: 1-swap moves
    selected = list(best_selected)
    current_score = best_score
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        selected_set = set(universe[i] for i in selected)
        
        for remove_idx in range(len(selected)):
            old_val = universe[remove_idx]
            # Try adding each value not currently in selected
            for add_val in universe:
                if add_val in selected_set:
                    continue
                if _swap_improvement(selected, remove_idx, add_val):
                    # Perform swap
                    selected[remove_idx] = add_val - 2  # index of add_val
                    selected_set.remove(old_val)
                    selected_set.add(add_val)
                    current_score += 1.0 / add_val - 1.0 / old_val
                    improved = True
                    break
            if improved:
                break
    
    # Convert back to values
    result = [universe[i] for i in selected]
    return result
# EVOLVE-BLOCK-END
