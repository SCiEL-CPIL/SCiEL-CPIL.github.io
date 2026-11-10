"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 256

def _precompute_4ap_conflicts():
    """For each x, return set of (a,d) pairs such that x is part of a 4-AP (a, a+d, a+2d, a+3d)."""
    conflicts = [set() for _ in range(N+1)]
    for a in range(1, N+1):
        for d in range(1, (N - a) // 3 + 1):
            x1, x2, x3, x4 = a, a+d, a+2*d, a+3*d
            # Any subset of 3 or more from {x1,x2,x3,x4} that includes x creates conflict
            # Actually: x conflicts if there exist a,d such that x is one of the 4 terms
            # and the other 3 are all in selected.
            # For incremental check: when adding x, we need to check if there's a 4-AP containing x
            # where the other 3 elements are already in selected.
            conflicts[x1].add((a, d, 1))  # x1 is position 1
            conflicts[x2].add((a, d, 2))  # x2 is position 2
            conflicts[x3].add((a, d, 3))  # x3 is position 3
            conflicts[x4].add((a, d, 4))  # x4 is position 4
    return conflicts

def _precompute_3ap_contributions():
    """For each pair (a, a+2d), the middle is a+d. Count 3-APs via middle element."""
    # A 3-AP is (a, a+d, a+2d). Count once per (a,d).
    # For incremental: adding x contributes 3-APs where x is first, middle, or last.
    pass

def _has_4ap_fast(selected_bool, x):
    """Check if adding x to selected_bool creates a 4-AP. selected_bool is a boolean array."""
    # Case 1: x = a (first term)
    # d such that x+3d <= N
    max_d1 = (N - x) // 3
    for d in range(1, max_d1 + 1):
        if selected_bool[x+d] and selected_bool[x+2*d] and selected_bool[x+3*d]:
            return True
        
    # Case 2: x = a+d (second term)
    # a = x-d. Need a>=1 => d<x. Need a+3d <= N => x+2d <= N => d <= (N-x)//2
    max_d2 = min(x - 1, (N - x) // 2)
    for d in range(1, max_d2 + 1):
        a = x - d
        if selected_bool[a] and selected_bool[x+d] and selected_bool[x+2*d]:
            return True
            
    # Case 3: x = a+2d (third term)
    # a = x-2d. Need a>=1 => 2d < x => d <= (x-1)//2. Need a+3d <= N => x+d <= N => d <= N-x
    max_d3 = min((x - 1) // 2, N - x)
    for d in range(1, max_d3 + 1):
        a = x - 2*d
        if selected_bool[a] and selected_bool[x-d] and selected_bool[x+d]:
            return True
            
    # Case 4: x = a+3d (fourth term)
    # a = x-3d. Need a>=1 => 3d < x => d <= (x-1)//3. Need a+3d <= N => x <= N (always true)
    max_d4 = (x - 1) // 3
    for d in range(1, max_d4 + 1):
        a = x - 3*d
        if selected_bool[a] and selected_bool[x-2*d] and selected_bool[x-d]:
            return True
            
    return False

def _count_3aps_fast(selected_bool):
    """Count 3-APs using boolean array."""
    count = 0
    for a in range(1, N+1):
        if not selected_bool[a]:
            continue
        for d in range(1, (N - a) // 2 + 1):
            if selected_bool[a+d] and selected_bool[a+2*d]:
                count += 1
    return count

def _count_3aps_delta(selected_bool, x):
    """Count 3-APs that involve x (as first, middle, or last term)."""
    delta = 0
    # x as first: (x, x+d, x+2d)
    max_d1 = (N - x) // 2
    for d in range(1, max_d1 + 1):
        if selected_bool[x+d] and selected_bool[x+2*d]:
            delta += 1
            
    # x as middle: (x-d, x, x+d)
    # Need x-d >= 1 => d <= x-1. Need x+d <= N => d <= N-x
    max_d2 = min(x - 1, N - x)
    for d in range(1, max_d2 + 1):
        if selected_bool[x-d] and selected_bool[x+d]:
            delta += 1
            
    # x as last: (x-2d, x-d, x)
    # Need x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2
    max_d3 = (x - 1) // 2
    for d in range(1, max_d3 + 1):
        if selected_bool[x-2*d] and selected_bool[x-d]:
            delta += 1
            
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s margin
    
    # Use numpy boolean array for fast membership
    selected_bool = np.zeros(N+1, dtype=bool)
    
    # Greedy initialization
    for x in range(1, N+1):
        if not _has_4ap_fast(selected_bool, x):
            selected_bool[x] = True
    
    current_score = _count_3aps_fast(selected_bool)
    best_selected = selected_bool.copy()
    best_score = current_score
    
    # Simulated annealing with incremental scoring
    random.seed(42)
    temperature = 15.0
    cooling = 0.9997
    min_temp = 0.05
    
    while time.time() < deadline and temperature > min_temp:
        move_type = random.random()
        
        if move_type < 0.5:
            # Add an element
            candidates = [x for x in range(1, N+1) if not selected_bool[x]]
            if not candidates:
                break
            x = random.choice(candidates)
            if not _has_4ap_fast(selected_bool, x):
                new_score = current_score + _count_3aps_delta(selected_bool, x)
                delta = new_score - current_score
                if delta >= 0 or random.random() < np.exp(delta / temperature):
                    selected_bool[x] = True
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = selected_bool.copy()
        elif move_type < 0.8:
            # Remove an element
            sel_list = [x for x in range(1, N+1) if selected_bool[x]]
            if not sel_list:
                break
            x = random.choice(sel_list)
            # Delta for removing x
            delta_remove = -_count_3aps_delta(selected_bool, x)
            new_score = current_score + delta_remove
            if delta_remove >= 0 or random.random() < np.exp(delta_remove / temperature):
                selected_bool[x] = False
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = selected_bool.copy()
        else:
            # Swap
            sel_list = [x for x in range(1, N+1) if selected_bool[x]]
            unselected_list = [x for x in range(1, N+1) if not selected_bool[x]]
            if not sel_list or not unselected_list:
                break
            x_out = random.choice(sel_list)
            x_in = random.choice(unselected_list)
            # Check if adding x_in is valid (after removing x_out)
            # Temporarily remove x_out
            selected_bool[x_out] = False
            if not _has_4ap_fast(selected_bool, x_in):
                # Compute delta: remove x_out, add x_in
                delta_remove = -_count_3aps_delta(selected_bool, x_out)
                delta_add = _count_3aps_delta(selected_bool, x_in)
                total_delta = delta_remove + delta_add
                new_score = current_score + total_delta
                if total_delta >= 0 or random.random() < np.exp(total_delta / temperature):
                    selected_bool[x_in] = True
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = selected_bool.copy()
                else:
                    # Revert: put x_out back
                    selected_bool[x_out] = True
            else:
                # Revert
                selected_bool[x_out] = True
        
        temperature *= cooling
    
    # Final local search: try to add any element
    for x in range(1, N+1):
        if not best_selected[x]:
            if not _has_4ap_fast(best_selected, x):
                best_selected[x] = True
                best_score = _count_3aps_fast(best_selected)
    
    # Convert to sorted list
    result = [x for x in range(1, N+1) if best_selected[x]]
    return sorted(result)
# EVOLVE-BLOCK-END
