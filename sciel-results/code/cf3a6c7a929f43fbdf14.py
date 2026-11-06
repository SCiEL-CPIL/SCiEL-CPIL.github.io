import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    points_list = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points_list)
    
    def get_idx(x, y):
        return x * N + y
    
    # Pre-compute squared distances
    xs = np.array([p[0] for p in points_list], dtype=np.int32)
    ys = np.array([p[1] for p in points_list], dtype=np.int32)
    
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    best_points = []
    best_count = 0
    
    # Helper to check if a point can be added to a set represented by indices
    # Returns (valid, d_to_selected_list) if valid, else (False, None)
    # d_to_selected_list is distances from new_idx to all sel_indices
    def try_add(new_idx, sel_indices, dist_sets):
        if not sel_indices:
            return True, []
        
        d_to_selected = dist_sq[new_idx, sel_indices]
        
        # Check 1: uniqueness in d_to_selected
        # We need to check if any distance appears twice.
        # Also check against dist_sets.
        
        # Optimization: If len(sel_indices) is small, set is fast.
        # If large, we can use a hash set.
        
        # Check duplicates in d_to_selected
        # Note: d_to_selected is a numpy array.
        # Converting to list and set is O(k).
        d_list = d_to_selected.tolist()
        if len(set(d_list)) != len(d_list):
            return False, None
        
        # Check against existing dist_sets
        for i, sel_idx in enumerate(sel_indices):
            d = d_list[i]
            if d in dist_sets[sel_idx]:
                return False, None
        
        return True, d_list

    def greedy_construct():
        indices = list(range(num_points))
        random.shuffle(indices)
        
        selected_indices = []
        dist_sets = []
        
        for idx in indices:
            valid, d_list = try_add(idx, selected_indices, dist_sets)
            if valid:
                selected_indices.append(idx)
                dist_sets.append(set(d_list))
                for i in range(len(selected_indices) - 1):
                    dist_sets[i].add(d_list[i])
                    
        return [points_list[i] for i in selected_indices]

    def simulated_annealing(initial_points):
        # Initialize state
        S = set()
        for p in initial_points:
            S.add(get_idx(p[0], p[1]))
        
        sel_indices = list(S)
        dist_sets = {}
        for i in sel_indices:
            ds = set()
            for j in sel_indices:
                if i != j:
                    ds.add(dist_sq[i, j])
            dist_sets[i] = ds
            
        current_size = len(S)
        best_size_sa = current_size
        best_points_sa = [points_list[i] for i in S]
        
        T = 5.0
        cooling_rate = 0.995
        min_T = 0.1
        
        # Precompute list of all indices for random sampling
        all_indices = list(range(num_points))
        
        while time.time() < deadline - 5 and T > min_T:
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random point
                new_idx = random.choice(all_indices)
                if new_idx in S:
                    continue
                
                valid, d_list = try_add(new_idx, sel_indices, dist_sets)
                if valid:
                    # Accept
                    S.add(new_idx)
                    sel_indices.append(new_idx)
                    dist_sets[new_idx] = set(d_list)
                    for i in range(len(sel_indices) - 1):
                        dist_sets[sel_indices[i]].add(d_list[i])
                    current_size += 1
                    if current_size > best_size_sa:
                        best_size_sa = current_size
                        best_points_sa = [points_list[i] for i in S]
                else:
                    # Reject (or accept with prob 0 since it's invalid)
                    pass
                    
            elif move_type < 0.7:
                # Try to remove a random point
                if not S:
                    continue
                rem_idx = random.choice(list(S))
                
                # Remove from S and update dist_sets
                S.remove(rem_idx)
                sel_indices.remove(rem_idx)
                for i in sel_indices:
                    dist_sets[i].remove(dist_sq[i, rem_idx])
                del dist_sets[rem_idx]
                
                # This move decreases size. 
                # In SA on valid sets, we usually don't want to decrease size unless stuck.
                # But here, removing allows us to add something else later.
                # We accept this move unconditionally? 
                # No, we should treat it as a step in the random walk.
                # Since it's valid, we accept it.
                current_size -= 1
                
            else:
                # Try to swap
                if not S:
                    continue
                rem_idx = random.choice(list(S))
                # Pick a candidate to add
                # Sample a few candidates to find a valid one
                swapped = False
                for _ in range(10):
                    new_idx = random.choice(all_indices)
                    if new_idx in S:
                        continue
                    
                    # Check if S \ {rem_idx} U {new_idx} is valid
                    # 1. Remove rem_idx temporarily
                    S.remove(rem_idx)
                    sel_indices.remove(rem_idx)
                    temp_dist_sets = {}
                    for i in sel_indices:
                        temp_dist_sets[i] = set(dist_sets[i])
                        temp_dist_sets[i].remove(dist_sq[i, rem_idx])
                    
                    # 2. Try to add new_idx
                    valid, d_list = try_add(new_idx, sel_indices, temp_dist_sets)
                    
                    if valid:
                        # Commit swap
                        S.add(new_idx)
                        sel_indices.append(new_idx)
                        dist_sets = temp_dist_sets
                        dist_sets[new_idx] = set(d_list)
                        for i in range(len(sel_indices) - 1):
                            dist_sets[sel_indices[i]].add(d_list[i])
                        swapped = True
                        break
                    else:
                        # Rollback
                        S.add(rem_idx)
                        sel_indices.append(rem_idx)
                        # Restore dist_sets (we modified temp_dist_sets, so original dist_sets is intact? 
                        # No, we created temp_dist_sets from dist_sets. 
                        # But we removed rem_idx from S and sel_indices. 
                        # We need to restore dist_sets for rem_idx?
                        # Actually, if we failed, we just need to put rem_idx back.
                        # The dist_sets for other points were not modified in the original dict, 
                        # because we copied them to temp_dist_sets.
                        # So we just need to restore S and sel_indices.
                        pass
                
                if swapped:
                    # Size remains same
                    pass
                else:
                    # Swap failed, state unchanged
                    pass
            
            T *= cooling_rate
            
        return best_points_sa

    # Main Loop
    # Run a few greedy restarts to get a good initial state for SA
    initial_candidates = []
    for _ in range(5):
        if time.time() > deadline - 100:
            break
        pts = greedy_construct()
        initial_candidates.append(pts)
        if len(pts) > best_count:
            best_count = len(pts)
            best_points = pts

    # Run SA on the best candidate found so far
    if best_points and time.time() < deadline - 20:
        improved = simulated_annealing(best_points)
        if len(improved) > best_count:
            best_count = len(improved)
            best_points = improved
            
    # If time remains, run SA on other good candidates or restart
    while time.time() < deadline - 5:
        # Pick a random candidate from initial_candidates or best
        base = random.choice(initial_candidates) if initial_candidates else best_points
        if not base:
            base = greedy_construct()
            
        improved = simulated_annealing(base)
        if len(improved) > best_count:
            best_count = len(improved)
            best_points = improved

    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
