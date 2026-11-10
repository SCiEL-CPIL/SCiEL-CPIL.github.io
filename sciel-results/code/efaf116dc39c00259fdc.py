import numpy as np
import random
import time

N = 64
TOTAL_POINTS = N * N

def construct():
    # Precompute coordinates
    coords = np.array([(x, y) for x in range(N) for y in range(N)])
    
    # Precompute squared distances
    # dist2[i, j] is squared distance between coords[i] and coords[j]
    # Use broadcasting
    # coords shape (4096, 2)
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # dist2 = np.sum(diff**2, axis=2)
    # This creates a 4096x4096 array, which is 16M floats. ~128MB. Acceptable.
    
    # To save memory, we can compute on the fly or use a smaller type.
    # Int16 is enough for N=64 (max dist 2*63^2 ~ 8000).
    
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Compute dist2 as int32 to be safe
    dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
    for i in range(TOTAL_POINTS):
        dx = x_coords[i] - x_coords
        dy = y_coords[i] - y_coords
        dist2[i] = dx*dx + dy*dy

    random.seed(42)
    
    def local_search(start_set, time_limit):
        current_set = list(start_set)
        in_set = set(current_set)
        
        # D[i] is the set of distances from current_set[i] to other points in current_set
        # We need to maintain this efficiently.
        # Instead of storing sets, we can recompute or use a dictionary of counts?
        # For small k, sets are fine.
        
        def get_dist_sets(s):
            ds = []
            for i, p in enumerate(s):
                d_set = set()
                for j, q in enumerate(s):
                    if i != j:
                        d_set.add(dist2[p][q])
                ds.append(d_set)
            return ds

        current_dist_sets = get_dist_sets(current_set)
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add a point
            # Iterate through points not in set
            # Optimization: Only check points that are "close" or random?
            # Let's just iterate all, but break early if invalid.
            
            added = False
            for idx in range(TOTAL_POINTS):
                if idx in in_set:
                    continue
                
                # Check distances from idx to current_set
                # d_vals = [dist2[idx][p] for p in current_set]
                d_vals = dist2[idx, current_set]
                
                # Check distinctness
                if len(d_vals) != len(set(d_vals)):
                    continue
                
                # Check conflicts with existing dist sets
                valid = True
                for i, p in enumerate(current_set):
                    if d_vals[i] in current_dist_sets[i]:
                        valid = False
                        break
                
                if valid:
                    # Add idx
                    current_set.append(idx)
                    in_set.add(idx)
                    current_dist_sets.append(set(d_vals))
                    
                    # Update existing dist sets
                    for i, p in enumerate(current_set[:-1]):
                        current_dist_sets[i].add(dist2[p][idx])
                        
                    improved = True
                    added = True
                    break # Restart add loop
            
            if not added:
                # 2. Try swaps
                # Pick a random point to remove
                if current_set:
                    found_swap = False
                    for _ in range(10): # Try a few random swaps
                        p_remove = random.choice(current_set)
                        pos = current_set.index(p_remove)
                        
                        # Temporarily remove
                        removed_set = current_set[:]
                        removed_set.pop(pos)
                        
                        # Rebuild dist sets for removed_set
                        temp_dist_sets = get_dist_sets(removed_set)
                        
                        # Try to add a random point
                        # Instead of random, try points that were close to p_remove?
                        # Or just random.
                        for _ in range(20):
                            q = random.randint(0, TOTAL_POINTS - 1)
                            if q in removed_set:
                                continue
                            
                            d_vals = dist2[q, removed_set]
                            if len(d_vals) != len(set(d_vals)):
                                continue
                            
                            valid = True
                            for i, p in enumerate(removed_set):
                                if d_vals[i] in temp_dist_sets[i]:
                                    valid = False
                                    break
                            
                            if valid:
                                # Accept swap
                                current_set = removed_set
                                current_set.append(q)
                                in_set = set(current_set)
                                current_dist_sets = get_dist_sets(current_set)
                                found_swap = True
                                improved = True
                                break
                            
                        if found_swap:
                            break
                    
                    if not found_swap:
                        # Perturbation: Remove a random point to escape local optimum
                        if len(current_set) > 2:
                            p_remove = random.choice(current_set)
                            pos = current_set.index(p_remove)
                            current_set.pop(pos)
                            in_set = set(current_set)
                            current_dist_sets = get_dist_sets(current_set)
                            improved = True # Force loop to continue

        return current_set

    # Initial Greedy
    best_set = []
    for _ in range(5):
        indices = list(range(TOTAL_POINTS))
        random.shuffle(indices)
        current_set = []
        in_set = set()
        current_dist_sets = []
        
        for idx in indices:
            if not current_set:
                current_set.append(idx)
                in_set.add(idx)
                current_dist_sets.append(set())
                continue
                
            d_vals = dist2[idx, current_set]
            if len(d_vals) != len(set(d_vals)):
                continue
            
            valid = True
            for i, p in enumerate(current_set):
                if d_vals[i] in current_dist_sets[i]:
                    valid = False
                    break
            
            if valid:
                current_set.append(idx)
                in_set.add(idx)
                current_dist_sets.append(set(d_vals))
                for i, p in enumerate(current_set[:-1]):
                    current_dist_sets[i].add(dist2[p][idx])
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]

    # Local Search from best_set
    final_set = local_search(best_set, 280)
    
    return [tuple(coords[i]) for i in final_set]
