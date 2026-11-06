import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Grid points
    xs = np.arange(N)
    ys = np.arange(N)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel()))
    num_points = N * N
    
    # Precompute squared distances
    # dist_sq[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    dx = points[:, 0:1] - points[1:, 0:1].T # This is wrong shape logic if not careful
    # Vectorized distance matrix
    # (N*N, 1, 1) - (1, N*N, 1) -> (N*N, N*N, 1)
    # Simpler:
    x = points[:, 0]
    y = points[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    # dist_sq is int32 or int64. Max dist is 63^2+63^2 ~ 8000. int16 is enough.
    dist_sq = dist_sq.astype(np.int16)
    
    # Helper to calculate cost of a set S (list of indices)
    # We maintain state incrementally.
    
    def greedy_random_restart():
        indices = list(range(num_points))
        random.shuffle(indices)
        
        selected = []
        # dist_counts[i] is a dict: {distance: count} for points in selected
        dist_counts = []
        # To speed up, we can store as list of dicts or arrays.
        # Since distances are small (< 10000), we could use arrays, but dict is fine for sparse sets.
        
        for idx in indices:
            # Check if idx can be added
            # 1. Distances from idx to all selected must be distinct
            # 2. For each selected i, d(idx, i) must not already exist in dist_counts[i]
            
            d_to_sel = dist_sq[idx, selected]
            
            if len(d_to_sel) > 0:
                # Check uniqueness in d_to_sel
                if len(set(d_to_sel.tolist())) != len(d_to_sel):
                    continue
                
                # Check against existing counts
                valid = True
                for i, sel_idx in enumerate(selected):
                    d = d_to_sel[i]
                    if d in dist_counts[i]:
                        valid = False
                        break
                
                if not valid:
                    continue
            
            # Accept
            selected.append(idx)
            # New dist_counts for idx
            new_counts = {}
            for d in d_to_sel:
                new_counts[d] = 1
            dist_counts.append(new_counts)
            
            # Update existing dist_counts
            for i, sel_idx in enumerate(selected[:-1]):
                d = d_to_sel[i]
                dist_counts[i][d] = dist_counts[i].get(d, 0) + 1
                
        return selected

    def calculate_cost(selected, dist_counts):
        # Cost = sum of violations
        # Violation for i: any distance with count > 1
        cost = 0
        for i in range(len(selected)):
            for d, c in dist_counts[i].items():
                if c > 1:
                    cost += (c - 1) # Or just 1? 
                    # Actually, if c=2, it forms 1 isosceles triangle with the 2 points?
                    # The constraint is "no three points". If there are 3 points at same dist, that's more violations.
                    # Let's just count the number of pairs of points at the same distance?
                    # No, just count "bad" distance values.
                    # If c=2, it's 1 violation type.
                    # Let's use cost = sum(c-1 for c>1).
        return cost

    # Simulated Annealing
    def simulated_annealing(initial_set):
        # Rebuild dist_counts
        selected = list(initial_set)
        dist_counts = []
        for i in selected:
            counts = {}
            for j in selected:
                if i != j:
                    d = dist_sq[i, j]
                    counts[d] = counts.get(d, 0) + 1
            dist_counts.append(counts)
        
        current_cost = calculate_cost(selected, dist_counts)
        best_set = list(selected) if current_cost == 0 else []
        best_size = len(best_set)
        
        T = 10.0
        cooling = 0.999
        
        # Precompute neighbors? No, random.
        
        while time.time() < deadline - 10 and T > 0.1:
            # Random Move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_selected = list(selected)
            new_dist_counts = [dict(c) for c in dist_counts] # Shallow copy of dicts
            
            if move_type == 'add':
                if len(selected) == num_points:
                    continue
                # Pick random not in selected
                # For efficiency, pick from random indices
                for _ in range(10):
                    cand = random.randint(0, num_points - 1)
                    if cand not in selected:
                        break
                else:
                    continue
                
                # Try to add cand
                # Calculate delta cost
                # Distances from cand to all in selected
                d_to_sel = dist_sq[cand, selected]
                
                # Check if it introduces violations
                # 1. Duplicates in d_to_sel
                # 2. Existing duplicates in dist_counts[i] for d_to_sel[i]
                
                # We need to simulate the addition
                # New counts for cand
                cand_counts = {}
                for d in d_to_sel:
                    cand_counts[d] = cand_counts.get(d, 0) + 1
                
                delta_cost = 0
                # Violations for cand
                for d, c in cand_counts.items():
                    if c > 1:
                        delta_cost += (c - 1)
                        
                # Violations for existing points
                for i, sel_idx in enumerate(selected):
                    d = d_to_sel[i]
                    old_c = dist_counts[i].get(d, 0)
                    new_c = old_c + 1
                    # Change in violation for i
                    # Old violation: max(0, old_c - 1)
                    # New violation: max(0, new_c - 1)
                    old_v = max(0, old_c - 1)
                    new_v = max(0, new_c - 1)
                    delta_cost += (new_v - old_v)
                    
                # Update state if we accept
                # For now, just calculate delta
                accept = False
                if delta_cost < 0:
                    accept = True
                elif delta_cost == 0:
                    accept = True
                else:
                    if random.random() < np.exp(-delta_cost / T):
                        accept = True
                
                if accept:
                    # Apply move
                    new_selected.append(cand)
                    new_dist_counts.append(cand_counts)
                    for i, sel_idx in enumerate(selected):
                        d = d_to_sel[i]
                        new_dist_counts[i][d] = new_dist_counts[i].get(d, 0) + 1
                    selected = new_selected
                    dist_counts = new_dist_counts
                    current_cost += delta_cost
                    if current_cost == 0 and len(selected) > best_size:
                        best_size = len(selected)
                        best_set = list(selected)
                        
            elif move_type == 'remove':
                if not selected:
                    continue
                idx_to_remove = random.randint(0, len(selected) - 1)
                cand = selected[idx_to_remove]
                
                # Calculate delta cost
                # Remove cand from selected
                # Distances from cand to others
                others = [s for i, s in enumerate(selected) if i != idx_to_remove]
                d_to_others = dist_sq[cand, others]
                
                delta_cost = 0
                # Violations for cand are removed
                for d, c in dist_counts[idx_to_remove].items():
                    delta_cost -= max(0, c - 1)
                    
                # Violations for others
                # Map from index in selected to index in dist_counts
                # We need to update dist_counts for others
                for i, sel_idx in enumerate(selected):
                    if i == idx_to_remove:
                        continue
                    d = dist_sq[sel_idx, cand]
                    old_c = dist_counts[i].get(d, 0)
                    new_c = old_c - 1
                    old_v = max(0, old_c - 1)
                    new_v = max(0, new_c - 1) if new_c > 0 else 0
                    delta_cost += (new_v - old_v)
                    
                # Accept if cost decreases or random
                accept = False
                if delta_cost < 0:
                    accept = True
                else:
                    if random.random() < np.exp(-delta_cost / T):
                        accept = True
                        
                if accept:
                    # Apply move
                    # Remove from selected
                    new_selected = [s for i, s in enumerate(selected) if i != idx_to_remove]
                    # Remove from dist_counts
                    new_dist_counts = []
                    for i, sel_idx in enumerate(selected):
                        if i == idx_to_remove:
                            continue
                        new_dc = dict(dist_counts[i])
                        d = dist_sq[sel_idx, cand]
                        new_dc[d] -= 1
                        if new_dc[d] == 0:
                            del new_dc[d]
                        new_dist_counts.append(new_dc)
                    selected = new_selected
                    dist_counts = new_dist_counts
                    current_cost += delta_cost
                    if current_cost == 0 and len(selected) > best_size:
                        best_size = len(selected)
                        best_set = list(selected)
                        
            elif move_type == 'swap':
                if len(selected) < 2 or len(selected) == num_points:
                    continue
                # Pick u in selected, v not in selected
                u_idx = random.randint(0, len(selected) - 1)
                u = selected[u_idx]
                
                v_found = False
                for _ in range(10):
                    v = random.randint(0, num_points - 1)
                    if v not in selected:
                        v_found = True
                        break
                if not v_found:
                    continue
                
                # Swap u and v
                # This is complex to calculate delta cost efficiently without full rebuild.
                # Given time constraints, skip swap in SA for now, rely on add/remove.
                # Or implement simple rebuild for swap if T is high?
                # Let's skip swap to keep code robust and fast.
                pass

            T *= cooling

        # Final cleanup: if best_set is empty, try to fix current state?
        # If current_cost > 0, we should remove points until valid.
        if current_cost > 0:
             # Greedy removal
             while current_cost > 0 and selected:
                 # Find point with most violations
                 max_viol = -1
                 max_idx = -1
                 for i in range(len(selected)):
                     v = sum(max(0, c-1) for c in dist_counts[i].values())
                     if v > max_viol:
                         max_viol = v
                         max_idx = i
                 if max_idx == -1: break
                 # Remove max_idx
                 cand = selected[max_idx]
                 others = [s for i, s in enumerate(selected) if i != max_idx]
                 # Update cost
                 # ... (Simplified: just remove and recalc cost if small)
                 # For speed, just remove and break if time is low
                 selected.pop(max_idx)
                 dist_counts.pop(max_idx)
                 # Recalc cost for remaining? Expensive.
                 # Just break and return best_set found so far.
                 break
                 
        return best_set

    # Main Loop
    best_points = []
    best_count = 0
    
    # Run multiple greedy restarts
    trials = 0
    while time.time() < deadline - 60:
        pts = greedy_random_restart()
        if len(pts) > best_count:
            best_count = len(pts)
            best_points = pts
            # Try SA on this best
            if best_count > 20 and time.time() < deadline - 30:
                improved = simulated_annealing(pts)
                if len(improved) > best_count:
                    best_count = len(improved)
                    best_points = improved
        trials += 1
        
    # Convert indices to points
    return [points[i] for i in best_points]
