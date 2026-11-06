"""No-isosceles set construction on [0, N)^2 grid."""
import time
import random
import math
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    # Pre-compute all grid points and indices
    all_points = [(x, y) for x in range(N) for y in range(N)]
    M = len(all_points)
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Pre-compute squared distance matrix
    # Shape (M, M)
    # Using broadcasting
    coords = np.array(all_points)
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # D = np.sum(diff**2, axis=-1)
    # This might be memory heavy if not careful, but M=4096 is okay.
    # 4096 * 4096 * 8 bytes = 128MB.
    
    # More memory efficient way?
    # Or just compute on the fly?
    # With NumPy, on-the-fly for a single candidate vs K points is fast.
    # But we do this for every candidate in greedy.
    # Precomputing D is better.
    
    # Let's try to compute D efficiently.
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # D[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    # We can compute this in blocks or just use the full matrix.
    # Let's use the full matrix.
    
    # To save memory, we can use int32 or float32?
    # Max dist squared is (63)^2 + (63)^2 = 3969 + 3969 = 7938.
    # Fits in uint16 or int16.
    # Let's use np.uint16 to save memory (64MB).
    
    dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    D = (dx * dx + dy * dy).astype(np.uint16)
    
    def is_valid_addition(candidate_idx, current_indices):
        """Check if adding candidate_idx to current_indices creates any isosceles triangle."""
        if not current_indices:
            return True
            
        cur_arr = np.array(current_indices, dtype=np.int64)
        n = len(cur_arr)
        
        # Distances from candidate to all current points
        d_c_s = D[candidate_idx, cur_arr]
        
        # Check for duplicates in d_c_s (vertex at candidate)
        # If any two distances are equal, isosceles triangle formed with candidate as apex
        if len(np.unique(d_c_s)) != n:
            return False
            
        # Distances between current points
        d_s_s = D[cur_arr][:, cur_arr]
        
        # Check if any distance from candidate to A equals distance between A and B
        # This corresponds to isosceles triangle with vertex at A (or B)
        # d_s_s[i, j] is dist(cur[i], cur[j])
        # d_c_s[i] is dist(candidate, cur[i])
        # We check if d_c_s[i] is in d_s_s[i, :]
        # i.e., is there j such that d_s_s[i, j] == d_c_s[i]?
        
        # d_c_s[:, None] has shape (n, 1)
        # d_s_s has shape (n, n)
        # Comparison gives (n, n) boolean matrix
        # If any True, then for some i, j: dist(c, cur[i]) == dist(cur[i], cur[j])
        # This forms triangle (c, cur[i], cur[j]) with sides dist(c, cur[i]) and dist(cur[i], cur[j]) equal.
        # This is isosceles.
        
        if (d_s_s == d_c_s[:, None]).any():
            return False
            
        return True

    def greedy_construct():
        current_set = []
        in_set = set()
        remaining = list(range(M))
        random.shuffle(remaining)
        
        for candidate_idx in remaining:
            if time.time() > deadline:
                break
            if candidate_idx in in_set:
                continue
            if is_valid_addition(candidate_idx, current_set):
                current_set.append(candidate_idx)
                in_set.add(candidate_idx)
                
        return current_set

    def local_search(initial_set, max_iterations=5000):
        current_set = list(initial_set)
        in_set = set(current_set)
        best_set = list(current_set)
        best_size = len(current_set)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to add a random point
            remaining_indices = [i for i in range(M) if i not in in_set]
            if remaining_indices:
                candidate_idx = random.choice(remaining_indices)
                if is_valid_addition(candidate_idx, current_set):
                    current_set.append(candidate_idx)
                    in_set.add(candidate_idx)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
            
            # Occasionally try to remove a point and add a different one
            if len(current_set) > 0 and random.random() < 0.3:
                idx_to_remove = random.randint(0, len(current_set) - 1)
                removed_idx = current_set.pop(idx_to_remove)
                in_set.discard(removed_idx)
                
                # Try to add a better point
                remaining_indices = [i for i in range(M) if i not in in_set]
                random.shuffle(remaining_indices)
                added = False
                for candidate_idx in remaining_indices[:100]:
                    if is_valid_addition(candidate_idx, current_set):
                        current_set.append(candidate_idx)
                        in_set.add(candidate_idx)
                        added = True
                        if len(current_set) > best_size:
                            best_set = list(current_set)
                            best_size = len(current_set)
                        break
                
                if not added:
                    # Put the removed point back
                    current_set.append(removed_idx)
                    in_set.add(removed_idx)
        
        return best_set

    best_overall = []
    best_size_overall = 0
    
    num_restarts = 20  # Increased restarts due to speedup
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 67890)
        
        current_set = greedy_construct()
        if len(current_set) > best_size_overall:
            best_overall = list(current_set)
            best_size_overall = len(current_set)
        
        if time.time() < deadline - 30:
            improved = local_search(current_set, max_iterations=2000)
            if len(improved) > best_size_overall:
                best_overall = list(improved)
                best_size_overall = len(improved)
    
    # Convert indices back to points
    return [all_points[i] for i in best_overall]
