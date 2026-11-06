import time
import random
import numpy as np

N = 64
GRID_SIZE = N * N

# Precompute points and distance matrix
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    best_set = []
    
    # Vectorized validity check
    def can_add(p_idx, selected_indices):
        k = len(selected_indices)
        if k == 0:
            return True
        if k == 1:
            return True # Always valid for 1 point
        
        # Get distances from p to selected points
        d_ps = DIST[p_idx, selected_indices]
        
        # Check 1: Distances from p to all selected must be unique
        if len(np.unique(d_ps)) != k:
            return False
            
        # Get distances between selected points
        # selected_indices is a numpy array of indices
        d_ss = DIST[np.ix_(selected_indices, selected_indices)]
        
        # Check 2: No isosceles triangle formed by p and any two points in S
        # d(a, b) == d(a, p)  OR  d(a, b) == d(b, p)
        # d_ss[i, j] == d_ps[i]  (Apex at a_i)
        # d_ss[i, j] == d_ps[j]  (Apex at a_j)
        
        # Broadcast d_ps to compare with d_ss
        # d_ps shape (k,), d_ss shape (k, k)
        # We want to check if d_ss[i, j] equals d_ps[i] or d_ps[j]
        
        # Mask out diagonal (i == j) as d_ss[i, i] = 0 and d_ps[i] > 0 (since p not in S)
        # Actually, if p is not in S, d_ps[i] > 0.
        
        conflict1 = (d_ss == d_ps[:, None])
        conflict2 = (d_ss == d_ps[None, :])
        
        # Exclude diagonal
        np.fill_diagonal(conflict1, False)
        np.fill_diagonal(conflict2, False)
        
        return not (np.any(conflict1) or np.any(conflict2))

    def greedy_init(order):
        selected = []
        for p_idx in order:
            if can_add(p_idx, np.array(selected)):
                selected.append(p_idx)
        return selected

    def local_search(selected, time_limit):
        # selected is a list of indices
        # We maintain a numpy array for fast indexing in can_add
        selected_arr = np.array(selected, dtype=np.int32)
        
        # Create a boolean mask for in-set points
        in_set = np.zeros(GRID_SIZE, dtype=bool)
        in_set[selected_arr] = True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(selected_arr)
            
            # Try "1-out, 2-in" moves
            # Shuffle the points in the set to try removals in random order
            remove_order = np.random.permutation(k)
            
            for r_idx in remove_order:
                if time.time() > time_limit:
                    break
                
                # Remove point r_idx
                remove_point = selected_arr[r_idx]
                new_selected = np.delete(selected_arr, r_idx)
                new_in_set = in_set.copy()
                new_in_set[remove_point] = False
                
                # Try to add 2 points
                # Get candidates from complement
                candidates = np.where(~new_in_set)[0]
                if len(candidates) < 2:
                    continue
                
                # Shuffle candidates to pick random ones
                np.random.shuffle(candidates)
                
                added_count = 0
                # Try first candidate
                c1 = candidates[0]
                if can_add(c1, new_selected):
                    new_selected = np.append(new_selected, c1)
                    new_in_set[c1] = True
                    added_count += 1
                    
                    # Try second candidate
                    # Exclude c1 from candidates
                    candidates2 = candidates[1:]
                    np.random.shuffle(candidates2)
                    for c2 in candidates2:
                        if can_add(c2, new_selected):
                            new_selected = np.append(new_selected, c2)
                            new_in_set[c2] = True
                            added_count += 1
                            break
                    
                # If we added at least 1 point, we improved or maintained size
                # But "1-out, 2-in" implies we want to add 2 to gain +1 net.
                # If we added 1, size is same. If 2, size is +1.
                # We accept if size increases. If size same, we might accept to escape local minima.
                # Let's accept if added_count >= 1.
                if added_count >= 1:
                    # Update state
                    selected_arr = new_selected
                    in_set = new_in_set
                    improved = True
                    # Break out of removal loop to restart with new set?
                    # Or continue? Restarting is safer to avoid cascading bad moves.
                    break
        
        return selected_arr

    # Main loop
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Create a random order
        order = np.random.permutation(GRID_SIZE)
        
        # Greedy initialization
        current_set = greedy_init(order)
        
        # Local search
        # Leave some time for next restart
        current_time = time.time()
        if current_time < deadline - 10:
            current_set = local_search(current_set, deadline - 10)
        
        # Update best
        if len(current_set) > len(best_set):
            best_set = current_set
            # Optional: print progress
            # print(f"Restart {num_restarts}: Found set of size {len(current_set)}")

    return [tuple(points_arr[i]) for i in best_set]
