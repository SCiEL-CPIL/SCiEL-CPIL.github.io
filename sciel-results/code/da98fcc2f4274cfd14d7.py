import numpy as np
import time
import random
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline, 40s margin
    
    num_cells = N * N
    
    # Precompute squared distances as int32 for fast hashing and memory efficiency
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    diff = coords[:, np.newaxis, :].astype(np.int32) - coords[np.newaxis, :, :].astype(np.int32)
    dist_sq = (diff[:, :, 0]**2 + diff[:, :, 1]**2).astype(np.int32)
    
    # Precompute for each cell, the list of (distance, neighbor) pairs sorted by distance
    # This helps with fast neighbor lookup
    # neighbor_info[i] = list of (d, j) for all j != i, sorted by d
    # We'll use this for structured initialization
    
    def get_neighbors_by_distance(i):
        """Return list of (distance, cell_index) for all cells, sorted by distance."""
        d = dist_sq[i, :].copy()
        d[i] = -1  # mark self
        order = np.argsort(d)
        return [(int(d[j]), int(j)) for j in order if d[j] >= 0]
    
    # Cache neighbor info lazily
    neighbor_cache = {}
    
    def get_cached_neighbors(i):
        if i not in neighbor_cache:
            neighbor_cache[i] = get_neighbors_by_distance(i)
        return neighbor_cache[i]
    
    # Incremental violation tracking structures
    # selected: set of selected cell indices
    # dist_counts[i]: dict {distance: count} for selected cells at that distance from i
    # violating: set of cells that have at least one duplicate distance
    # num_violating: count of violating cells
    
    def make_incremental_tracker():
        selected = set()
        dist_counts = [None] * num_cells
        num_violating = 0
        violating = set()
        
        def add_cell(j):
            nonlocal num_violating
            # j's perspective: count distances to all currently selected cells
            j_dists = {}
            for i in selected:
                d = int(dist_sq[j, i])
                j_dists[d] = j_dists.get(d, 0) + 1
            
            j_now_violating = any(c >= 2 for c in j_dists.values())
            
            # Update all existing selected cells' perspective
            for i in list(selected):
                d = int(dist_sq[i, j])
                if dist_counts[i] is None:
                    dist_counts[i] = {}
                old_count = dist_counts[i].get(d, 0)
                dist_counts[i][d] = old_count + 1
                new_count = old_count + 1
                
                if old_count == 1 and new_count == 2:
                    if i not in violating:
                        violating.add(i)
                        num_violating += 1
            
            selected.add(j)
            dist_counts[j] = j_dists
            
            if j_now_violating:
                violating.add(j)
                num_violating += 1
        
        def remove_cell(j):
            nonlocal num_violating
            # Update j's own status
            j_dists = dist_counts[j]
            if j_dists is not None:
                j_was_violating = any(c >= 2 for c in j_dists.values())
                if j_was_violating and j in violating:
                    violating.discard(j)
                    num_violating -= 1
            
            # Update all other selected cells
            for i in list(selected):
                if i == j:
                    continue
                d = int(dist_sq[i, j])
                if dist_counts[i] is not None:
                    old_count = dist_counts[i].get(d, 0)
                    if old_count > 0:
                        dist_counts[i][d] = old_count - 1
                        if old_count == 2:
                            # i might no longer be violating
                            new_violating = any(c >= 2 for c in dist_counts[i].values())
                            if not new_violating and i in violating:
                                violating.discard(i)
                                num_violating -= 1
            
            selected.discard(j)
            dist_counts[j] = None
        
        def is_valid():
            return num_violating == 0
        
        def objective(penalty_weight):
            return len(selected) - penalty_weight * num_violating
        
        return {
            'selected': selected,
            'dist_counts': dist_counts,
            'num_violating': lambda: num_violating,
            'violating': violating,
            'add_cell': add_cell,
            'remove_cell': remove_cell,
            'is_valid': is_valid,
            'objective': objective,
        }
    
    def structured_initialization(seed_val):
        """
        Structured initialization: start with a sparse grid pattern,
        then greedily add points in order of 'safety' (fewest conflicts).
        This gives a better starting point than pure random greedy.
        """
        random.seed(seed_val)
        tracker = make_incremental_tracker()
        selected = tracker['selected']
        
        # Phase 1: Place points on a coarse grid (every k cells)
        # This creates a structured base that's likely valid
        k = 4  # grid spacing
        grid_points = []
        for x in range(0, N, k):
            for y in range(0, N, k):
                idx = x * N + y
                grid_points.append(idx)
        
        random.shuffle(grid_points)
        
        # Add grid points greedily (only if valid)
        for idx in grid_points:
            if time.time() > deadline - 60:
                break
            # Check if adding idx would create violations
            # Quick check: for each selected i, see if dist_sq[idx, i] already exists in dist_counts[i]
            can_add = True
            for i in list(selected):
                d = int(dist_sq[idx, i])
                if tracker['dist_counts'][i] is not None:
                    if tracker['dist_counts'][i].get(d, 0) >= 1:
                        can_add = False
                        break
            
            if can_add:
                # Also check idx's own perspective
                idx_dists = {}
                for i in selected:
                    d = int(dist_sq[idx, i])
                    idx_dists[d] = idx_dists.get(d, 0) + 1
                if not any(c >= 2 for c in idx_dists.values()):
                    tracker['add_cell'](idx)
        
        # Phase 2: Greedy fill - add points in order of fewest potential conflicts
        remaining = [i for i in range(num_cells) if i not in selected]
        random.shuffle(remaining)
        
        # Score each remaining point by how many conflicts it would create
        # For efficiency, just try adding in random order
        for idx in remaining:
            if time.time() > deadline - 45:
                break
            
            # Quick validity check
            can_add = True
            for i in list(selected):
                d = int(dist_sq[idx, i])
                if tracker['dist_counts'][i] is not None:
                    if tracker['dist_counts'][i].get(d, 0) >= 1:
                        can_add = False
                        break
            
            if can_add:
                idx_dists = {}
                for i in selected:
                    d = int(dist_sq[idx, i])
                    idx_dists[d] = idx_dists.get(d, 0) + 1
                if not any(c >= 2 for c in idx_dists.values()):
                    tracker['add_cell'](idx)
        
        return tracker
    
    def swap_local_search(tracker, time_limit, seed_val):
        """
        Penalty-based local search with add/remove/swap moves.
        Uses incremental violation tracking to allow temporary violations,
        enabling escape from local optima.
        """
        random.seed(seed_val)
        selected = tracker['selected']
        num_cells_local = num_cells
        
        # Track best valid solution
        best_valid_count = len(selected) if tracker['is_valid']() else 0
        best_valid_selected = set(selected) if tracker['is_valid']() else set()
        
        # Current objective
        penalty_weight = 3.0
        current_obj = tracker['objective'](penalty_weight)
        best_obj = current_obj
        best_selected_snapshot = set(selected)
        
        T = 5.0
        T_min = 0.05
        cooling_rate = 0.9998
        
        start_ls = time.time()
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() - start_ls < time_limit and no_improve < max_no_improve:
            if time.time() > deadline - 15:
                break
            
            # Choose operation: add (40%), swap (40%), remove (20%)
            r = random.random()
            
            if r < 0.4:
                # Add a random unselected cell
                idx = random.randint(0, num_cells_local - 1)
                if idx in selected:
                    no_improve += 1
                    continue
                
                # Compute delta in objective
                # Adding idx: 
                # - idx's new violation status
                # - for each selected i, count change
                idx_dists = {}
                for i in selected:
                    d = int(dist_sq[idx, i])
                    idx_dists[d] = idx_dists.get(d, 0) + 1
                
                idx_new_violating = any(c >= 2 for c in idx_dists.values())
                
                new_i_violations = 0
                for i in selected:
                    d = int(dist_sq[i, idx])
                    if tracker['dist_counts'][i] is not None:
                        old_c = tracker['dist_counts'][i].get(d, 0)
                        if old_c == 1:
                            new_i_violations += 1
                
                delta_violations = (1 if idx_new_violating else 0) + new_i_violations
                delta_obj = 1.0 - penalty_weight * delta_violations
                
                # Accept if improvement, or with probability exp(-delta_obj/T) if worsening
                if delta_obj >= 0:
                    tracker['add_cell'](idx)
                    new_obj = tracker['objective'](penalty_weight)
                    no_improve = 0
                    if new_obj > best_obj:
                        best_obj = new_obj
                        best_selected_snapshot = set(selected)
                        if tracker['is_valid']():
                            if len(selected) > best_valid_count:
                                best_valid_count = len(selected)
                                best_valid_selected = set(selected)
                else:
                    if random.random() < np.exp(-delta_obj / max(T, 1e-10)):
                        tracker['add_cell'](idx)
                        no_improve = 0
                    else:
                        no_improve += 1
                        
            elif r < 0.8:
                # Swap: remove one selected, add one unselected
                if not selected:
                    no_improve += 1
                    continue
                
                old_idx = random.choice(list(selected))
                new_idx = random.randint(0, num_cells_local - 1)
                if new_idx in selected:
                    no_improve += 1
                    continue
                
                # Compute delta for removing old_idx and adding new_idx
                # This is complex; approximate by doing the operations and checking
                # For efficiency, just try it and revert if bad
                
                # Save state
                old_selected = set(selected)
                
                # Remove old_idx
                tracker['remove_cell'](old_idx)
                
                # Try to add new_idx
                can_add = True
                for i in list(selected):
                    d = int(dist_sq[new_idx, i])
                    if tracker['dist_counts'][i] is not None:
                        if tracker['dist_counts'][i].get(d, 0) >= 1:
                            # This would create a violation, but we allow it
                            pass
                
                # Just add it (violations allowed)
                tracker['add_cell'](new_idx)
                
                new_obj = tracker['objective'](penalty_weight)
                delta_obj = new_obj - current_obj
                
                if delta_obj >= 0 or random.random() < np.exp(-delta_obj / max(T, 1e-10)):
                    current_obj = new_obj
                    no_improve = 0
                    if new_obj > best_obj:
                        best_obj = new_obj
                        best_selected_snapshot = set(selected)
                        if tracker['is_valid']():
                            if len(selected) > best_valid_count:
                                best_valid_count = len(selected)
                                best_valid_selected = set(selected)
                else:
                    # Revert
                    tracker['remove_cell'](new_idx)
                    tracker['add_cell'](old_idx)
                    no_improve += 1
                    
            else:
                # Remove a random selected cell
                if not selected:
                    no_improve += 1
                    continue
                
                idx = random.choice(list(selected))
                tracker['remove_cell'](idx)
                
                new_obj = tracker['objective'](penalty_weight)
                delta_obj = new_obj - current_obj
                
                # Removal always reduces count by 1, but may reduce violations
                # Accept if it doesn't make things too bad
                if delta_obj >= -2.0 or random.random() < np.exp(-delta_obj / max(T, 1e-10)):
                    current_obj = new_obj
                    no_improve = 0
                    if new_obj > best_obj:
                        best_obj = new_obj
                        best_selected_snapshot = set(selected)
                        if tracker['is_valid']():
                            if len(selected) > best_valid_count:
                                best_valid_count = len(selected)
                                best_valid_selected = set(selected)
                else:
                    tracker['add_cell'](idx)
                    no_improve += 1
            
            # Cool down
            T = max(T * cooling_rate, T_min)
            
            # Periodically reset to best if stuck
            if no_improve > max_no_improve // 2:
                # Restore best snapshot
                for i in list(selected):
                    tracker['remove_cell'](i)
                for i in best_selected_snapshot:
                    tracker['add_cell'](i)
                current_obj = tracker['objective'](penalty_weight)
                no_improve = 0
                T = max(T, 3.0)
        
        # Restore best valid solution if we found one
        if best_valid_count > len(selected):
            for i in list(selected):
                tracker['remove_cell'](i)
            for i in best_valid_selected:
                tracker['add_cell'](i)
        
        return tracker
    
    # Main search loop
    best_points = []
    best_count = 0
    
    trial_count = 0
    max_trials = 15
    
    while time.time() < deadline - 20 and trial_count < max_trials:
        trial_count += 1
        seed_val = trial_count * 7919 + 12345
        
        # Structured initialization
        tracker = structured_initialization(seed_val)
        current_count = len(tracker['selected'])
        
        if current_count > best_count:
            best_count = current_count
            best_points = [(i // N, i % N) for i in tracker['selected']]
        
        # Apply penalty-based local search
        if time.time() < deadline - 25:
            time_for_ls = min(15.0, (deadline - 25) - time.time())
            if time_for_ls > 3.0:
                tracker_ls = swap_local_search(tracker, time_for_ls, seed_val + 999)
                ls_count = len(tracker_ls['selected'])
                
                if tracker_ls['is_valid']() and ls_count > best_count:
                    best_count = ls_count
                    best_points = [(i // N, i % N) for i in tracker_ls['selected']]
        
        # Early exit
        if time.time() > deadline - 30:
            break
    
    # Final validation: ensure no isosceles triangles
    if best_points:
        # Verify the solution is valid
        pts = set(best_points)
        valid = True
        pts_list = list(pts)
        n_pts = len(pts_list)
        
        # For each pair, check that no third point forms isosceles
        # This is O(n^3) but n is small enough
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                xi, yi = pts_list[i]
                xj, yj = pts_list[j]
                d_ij_sq = (xi - xj)**2 + (yi - yj)**2
                
                # For each other point k, check if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk
                for k in range(n_pts):
                    if k == i or k == j:
                        continue
                    xk, yk = pts_list[k]
                    d_ik_sq = (xi - xk)**2 + (yi - yk)**2
                    d_jk_sq = (xj - xk)**2 + (yj - yk)**2
                    
                    if d_ik_sq == d_ij_sq or d_jk_sq == d_ij_sq or d_ik_sq == d_jk_sq:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: rebuild from scratch with strict validity
            best_points = []
            best_count = 0
            random.seed(42)
            S = []
            for x in range(N):
                for y in range(N):
                    p = (x, y)
                    can_add = True
                    for q in S:
                        d_sq = (x - q[0])**2 + (y - q[1])**2
                        for r in S:
                            if r == q:
                                continue
                            d_qr_sq = (q[0] - r[0])**2 + (q[1] - r[1])**2
                            d_pr_sq = (x - r[0])**2 + (y - r[1])**2
                            if d_sq == d_qr_sq or d_sq == d_pr_sq or d_qr_sq == d_pr_sq:
                                can_add = False
                                break
                        if not can_add:
                            break
                    if can_add:
                        S.append(p)
            best_points = S
            best_count = len(S)
    
    return best_points
