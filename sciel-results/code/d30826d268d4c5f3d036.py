"""No-isosceles set: find max points in [0,N)^2 with no three forming an isosceles triangle."""
import time
import random
import math
import numpy as np

N = 64
TIME_LIMIT = 280  # Safety margin below 330s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N
    
    # Precompute all pairwise squared distances as a numpy array for speed
    # dist2[i, j] = squared distance between point i and point j
    coords = np.array(all_points, dtype=np.int32)
    # dist2 shape: (num_points, num_points)
    dx = coords[:, 0:1] - coords[1:0, 0]  # This is wrong, let's do it properly
    dx = coords[:, 0:1] - coords[:, 0:1].T
    dy = coords[:, 1:2] - coords[:, 1:2].T
    dist2 = (dx * dx + dy * dy).astype(np.int64)
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(p_idx, selected_indices, dist_from_sets):
        """
        Check if point p_idx can be added to selected_indices.
        selected_indices: list of indices of selected points
        dist_from_sets: list of sets, where dist_from_sets[i] contains squared distances 
                        from selected_indices[i] to all other selected points.
        Returns (valid, distances_to_selected)
        """
        p = p_idx
        # Check for duplicate distances from p to any selected point
        d_p = []
        for i in selected_indices:
            d = dist2[p, i]
            d_p.append(d)
            
        # If any two distances are equal, p forms an isosceles triangle with those two points
        if len(set(d_p)) != len(d_p):
            return False, None
            
        # Check if p creates an isosceles triangle with any pair of existing points
        # For each existing point i, check if d(p, i) is in dist_from_sets[i]
        for i, sel_idx in enumerate(selected_indices):
            if d_p[i] in dist_from_sets[sel_idx]:
                return False, None
        
        return True, d_p

    def add_point(p_idx, selected_indices, dist_from_sets, d_p):
        """Add point p_idx to the selection, updating data structures."""
        selected_indices.append(p_idx)
        # Create new distance set for p_idx
        new_dist_set = set()
        for i, sel_idx in enumerate(selected_indices[:-1]):
            new_dist_set.add(d_p[i])
        dist_from_sets.append(new_dist_set)
        # Update distance sets for all existing points
        for i, sel_idx in enumerate(selected_indices[:-1]):
            dist_from_sets[sel_idx].add(d_p[i])

    def rebuild_dist_from_sets(selected_indices):
        """Rebuild dist_from_sets from scratch for a given selection."""
        dist_from_sets = [set() for _ in range(num_points)]
        for i, idx_i in enumerate(selected_indices):
            for j, idx_j in enumerate(selected_indices):
                if i != j:
                    d = dist2[idx_i, idx_j]
                    dist_from_sets[idx_i].add(d)
        return dist_from_sets

    def greedy_select(order):
        """Greedy construction using a specific ordering of point indices."""
        selected_indices = []
        dist_from_sets = [set() for _ in range(num_points)]
        
        for p_idx in order:
            valid, d_p = is_valid_addition(p_idx, selected_indices, dist_from_sets)
            if valid:
                add_point(p_idx, selected_indices, dist_from_sets, d_p)
                
        return selected_indices

    def local_improvement(selected_indices, rng, max_iters=500):
        """
        Try to improve the set by adding points or swapping.
        Returns improved list of indices.
        """
        selected_indices = list(selected_indices)
        dist_from_sets = rebuild_dist_from_sets(selected_indices)
        
        # Build a set of remaining indices for quick lookup
        selected_set = set(selected_indices)
        remaining = [i for i in range(num_points) if i not in selected_set]
        rng.shuffle(remaining)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try to add a point from remaining
            for i, p_idx in enumerate(remaining):
                valid, d_p = is_valid_addition(p_idx, selected_indices, dist_from_sets)
                if valid:
                    add_point(p_idx, selected_indices, dist_from_sets, d_p)
                    selected_set.add(p_idx)
                    remaining.pop(i)
                    improved = True
                    break
            
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return selected_indices

    def swap_improvement(selected_indices, rng, max_swaps=500):
        """
        Try to improve by swapping points: remove one, try to add another.
        """
        selected_indices = list(selected_indices)
        dist_from_sets = rebuild_dist_from_sets(selected_indices)
        selected_set = set(selected_indices)
        
        for _ in range(max_swaps):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            if not selected_indices:
                break
                
            # Pick a random point to remove
            idx_to_remove = rng.randint(0, len(selected_indices) - 1)
            removed_idx = selected_indices[idx_to_remove]
            
            # Remove it
            selected_indices.pop(idx_to_remove)
            selected_set.discard(removed_idx)
            dist_from_sets = rebuild_dist_from_sets(selected_indices)
            
            # Try to add a random point from remaining
            remaining = [i for i in range(num_points) if i not in selected_set]
            if not remaining:
                # Re-add the removed point
                selected_indices.append(removed_idx)
                selected_set.add(removed_idx)
                dist_from_sets = rebuild_dist_from_sets(selected_indices)
                continue
                
            # Try a few random candidates
            found_add = False
            for _ in range(10):
                p_idx = rng.choice(remaining)
                valid, d_p = is_valid_addition(p_idx, selected_indices, dist_from_sets)
                if valid:
                    add_point(p_idx, selected_indices, dist_from_sets, d_p)
                    selected_set.add(p_idx)
                    found_add = True
                    break
            
            if not found_add:
                # Re-add the removed point
                selected_indices.append(removed_idx)
                selected_set.add(removed_idx)
                dist_from_sets = rebuild_dist_from_sets(selected_indices)
        
        return selected_indices

    # Multiple restarts with different orderings
    num_restarts = 30
    rng_master = random.Random(12345)
    
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        rng = random.Random(1000 + r * 7919)
        
        # Random ordering
        order = list(range(num_points))
        rng.shuffle(order)
        
        current_indices = greedy_select(order)
        current_size = len(current_indices)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_indices)
        
        # Local improvement (add points)
        if current_size >= best_size:
            improved_indices = local_improvement(current_indices, rng, max_iters=200)
            if len(improved_indices) > best_size:
                best_size = len(improved_indices)
                best_set = list(improved_indices)
        
        # Swap improvement
        if time.time() - start_time < TIME_LIMIT - 15:
            swapped_indices = swap_improvement(improved_indices if len(improved_indices) > current_size else current_indices, rng, max_swaps=200)
            if len(swapped_indices) > best_size:
                best_size = len(swapped_indices)
                best_set = list(swapped_indices)
    
    # Final local improvement on the best set found
    if time.time() - start_time < TIME_LIMIT - 10 and best_size > 0:
        final_rng = random.Random(99999)
        final_improved = local_improvement(best_set, final_rng, max_iters=1000)
        if len(final_improved) > best_size:
            best_size = len(final_improved)
            best_set = list(final_improved)
    
    # Convert indices back to points
    return [all_points[i] for i in best_set]
# EVOLVE-BLOCK-END
