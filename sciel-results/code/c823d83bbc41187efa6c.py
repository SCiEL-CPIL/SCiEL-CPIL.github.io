import random
import numpy as np
import time

N = 64

def construct():
    # Precompute all grid points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = len(points)
    
    best_size = 0
    best_set = []
    
    start_time = time.time()
    time_limit = 290  # Leave 10s buffer
    
    restart = 0
    while time.time() - start_time < time_limit:
        restart += 1
        # Shuffle points
        indices = np.arange(num_points)
        np.random.shuffle(indices)
        
        current_set = []
        # forbidden[i] will be a set of squared distances from current_set[i] to other selected points
        forbidden = []
        
        # We maintain a boolean mask of available points
        available = np.ones(num_points, dtype=bool)
        
        # Greedy loop
        while True:
            n_sel = len(current_set)
            
            if n_sel == 0:
                avail_indices = np.where(available)[0]
                if len(avail_indices) == 0:
                    break
                chosen_idx = avail_indices[np.random.randint(len(avail_indices))]
                
            elif n_sel == 1:
                avail_indices = np.where(available)[0]
                if len(avail_indices) == 0:
                    break
                chosen_idx = avail_indices[np.random.randint(len(avail_indices))]
                
            else:
                avail_indices = np.where(available)[0]
                if len(avail_indices) == 0:
                    break
                
                avail_pts = points[avail_indices]
                sel_pts = np.array(current_set) # Shape (n_sel, 2)
                
                dx = avail_pts[:, 0, None] - sel_pts[None, :, 0]
                dy = avail_pts[:, 1, None] - sel_pts[None, :, 1]
                dists = dx*dx + dy*dy
                
                # Check 1: No duplicate distances in any row
                sorted_dists = np.sort(dists, axis=1)
                has_dup = np.any(sorted_dists[:, 1:] == sorted_dists[:, :-1], axis=1)
                valid_mask = ~has_dup
                
                # Check 2: Forbidden distances
                for i in range(n_sel):
                    if not valid_mask.any():
                        break
                    col = dists[:, i]
                    forb_arr = np.array(list(forbidden[i]), dtype=np.int64)
                    is_forbidden = np.isin(col, forb_arr)
                    valid_mask &= ~is_forbidden
                
                if not valid_mask.any():
                    break
                
                valid_indices_in_avail = np.where(valid_mask)[0]
                chosen_local_idx = valid_indices_in_avail[np.random.randint(len(valid_indices_in_avail))]
                chosen_idx = avail_indices[chosen_local_idx]
            
            # Update state
            current_set.append(points[chosen_idx].tolist())
            available[chosen_idx] = False
            
            if n_sel > 0:
                new_pt = points[chosen_idx]
                sel_pts_arr = np.array(current_set[:-1])
                dx = sel_pts_arr[:, 0] - new_pt[0]
                dy = sel_pts_arr[:, 1] - new_pt[1]
                new_dists = dx*dx + dy*dy
                
                for i in range(n_sel):
                    forbidden[i].add(int(new_dists[i]))
                
                forbidden.append(set(new_dists.tolist()))
            else:
                forbidden.append(set())
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = [tuple(p) for p in current_set]
                
    return best_set
