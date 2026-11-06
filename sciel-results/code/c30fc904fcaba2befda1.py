import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1
    
    # Filter feasible
    mask = w <= cap
    if not np.any(mask):
        return -1
    w = w[mask]
    v = v[mask]
    n = len(w)
    
    # Sort by density descending
    densities = v / np.maximum(w, 1e-12)
    order = np.argsort(-densities)
    w = w[order]
    v = v[order]
    
    feasible_indices = np.where(mask)[0]
    orig_ids = feasible_indices[order]
    
    # B&B with tight time guard
    start_time = time.monotonic()
    deadline = start_time + 100.0  # Leave 50s safety margin for 150s limit
    
    best_solution = []
    best_val = 0.0
    nodes_visited = 0
    
    def bb(idx, cap_cur, val_cur, current_path):
        nonlocal best_val, best_solution, nodes_visited
        
        # Time check - hard deadline
        if time.monotonic() > deadline:
            return
        
        nodes_visited += 1
        
        if idx == n:
            if val_cur > best_val:
                best_val = val_cur
                best_solution = current_path[:]
            return
        
        # Tighter bound: fractional knapsack from idx
        b_val = val_cur
        b_cap = cap_cur
        for j in range(idx, n):
            if w[j] <= b_cap:
                b_val += v[j]
                b_cap -= w[j]
            else:
                b_val += v[j] * (b_cap / w[j])
                break
        
        if b_val <= best_val + 1e-12:
            return
        
        # Branch: Take item idx (prefer taking to find good solutions early)
        if w[idx] <= cap_cur:
            current_path.append(idx)
            bb(idx + 1, cap_cur - w[idx], val_cur + v[idx], current_path)
            current_path.pop()
        
        # Check time before skip branch
        if time.monotonic() > deadline:
            return
        
        # Branch: Skip item idx
        bb(idx + 1, cap_cur, val_cur, current_path)
    
    bb(0, cap, 0.0, [])
    
    # Check if we timed out
    if time.monotonic() > deadline:
        # Fallback: use greedy solution
        d = v / np.maximum(w, 1e-12)
        best_sorted_idx = int(np.argmax(d))
        return int(orig_ids[best_sorted_idx])
    
    if not best_solution:
        # Fallback to greedy if B&B found nothing
        d = v / np.maximum(w, 1e-12)
        best_sorted_idx = int(np.argmax(d))
        return int(orig_ids[best_sorted_idx])
    
    # Local refinement: try swapping items in the solution
    # Convert solution to set for O(1) lookup
    sol_set = set(best_solution)
    
    # Try removing each item and adding the best non-solution item that fits
    # This is a 1-swap local search
    improved = False
    for _ in range(3):  # Up to 3 refinement passes
        improved = False
        for remove_idx in best_solution[:]:
            new_cap = cap - sum(w[i] for i in best_solution if i != remove_idx)
            new_val = best_val - v[remove_idx]
            
            # Find best item not in solution that fits in new_cap
            best_add = -1
            best_add_val = -1
            for add_idx in range(n):
                if add_idx in sol_set or add_idx == remove_idx:
                    continue
                if w[add_idx] <= new_cap:
                    if v[add_idx] > best_add_val:
                        best_add_val = v[add_idx]
                        best_add = add_idx
            
            if best_add != -1 and new_val + best_add_val > best_val + 1e-12:
                # Apply swap
                best_solution.remove(remove_idx)
                best_solution.append(best_add)
                sol_set.discard(remove_idx)
                sol_set.add(best_add)
                best_val = new_val + best_add_val
                improved = True
                break
        
        if not improved:
            break
    
    # Pick the most critical item from the solution
    # Criticality: value contribution per unit weight, weighted by how much capacity it uses
    # Simpler: pick item with highest value (locks in value quickly)
    # Or: pick item whose removal would cause the largest value drop
    # Let's use: pick item with highest value/weight ratio among solution items
    # Actually, let's just pick the highest value item for simplicity and consistency
    
    sol_indices_in_sorted = best_solution
    vals = v[sol_indices_in_sorted]
    max_pos = int(np.argmax(vals))
    chosen_sorted_idx = sol_indices_in_sorted[max_pos]
    
    return int(orig_ids[chosen_sorted_idx])
