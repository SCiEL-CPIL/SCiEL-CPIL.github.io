import numpy as np
import random
import time
import math

N = 64
GRID_SIZE = N * N

def construct():
    start_time = time.time()
    deadline = 320  # 10s margin
    
    # 1. Pre-compute grid points and distance matrix
    # Points are (x, y) for x in [0, N), y in [0, N)
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    points = list(zip(xs, ys))
    
    # Distance matrix D[i, j] = squared distance
    # To save memory, we can compute on the fly or use a sparse representation?
    # 4096x4096 float32 is ~64MB. Acceptable.
    # But we can also just compute dx, dy arrays.
    
    # Let's use a function to get dist2
    # Actually, pre-computing the full matrix is faster for repeated lookups.
    # D[i, j]
    # We can store it as a 2D numpy array of int32.
    
    # Optimization: We don't need the full matrix if we just index by point index.
    # Let's create the matrix.
    x_coords = np.array(xs)
    y_coords = np.array(ys)
    
    # dx = x_i - x_j
    # This is (4096, 4096)
    # dx = x_coords[:, None] - x_coords[None, :]
    # dy = y_coords[:, None] - y_coords[None, :]
    # D = dx**2 + dy**2
    
    # This might take a second or two to compute.
    dx = x_coords[:, None] - x_coords[None, :]
    dy = y_coords[:, None] - y_coords[None, :]
    D = (dx**2 + dy**2).astype(np.int32)
    
    # 2. Greedy Initialization
    # We want to start with a reasonably large valid set.
    
    def greedy_init():
        s_indices = []
        dist_sets = {} # point_idx -> set of dist2
        
        # Shuffle indices
        order = list(range(GRID_SIZE))
        random.shuffle(order)
        
        for idx in order:
            # Check if idx can be added
            valid = True
            d_vals = []
            for s_idx in s_indices:
                d2 = D[idx, s_idx]
                d_vals.append(d2)
                
            # Check distinctness among d_vals
            if len(set(d_vals)) != len(d_vals):
                continue
                
            # Check against existing dist_sets
            for d2, s_idx in zip(d_vals, s_indices):
                if d2 in dist_sets[s_idx]:
                    valid = False
                    break
                    
            if valid:
                s_indices.append(idx)
                dist_sets[idx] = set(d_vals)
                for d2, s_idx in zip(d_vals, s_indices):
                    # Note: s_indices now includes idx, so we should iterate over original s_indices
                    # Actually, I appended idx to s_indices.
                    # The loop above iterated over the OLD s_indices.
                    # So I need to update the OLD ones.
                    pass
                
                # Let's re-do the update logic carefully
                # The check loop used the list BEFORE adding idx.
                # So:
                new_idx = idx
                old_indices = s_indices[:-1]
                old_d_vals = d_vals
                
                dist_sets[new_idx] = set(old_d_vals)
                for d2, s_idx in zip(old_d_vals, old_indices):
                    dist_sets[s_idx].add(d2)
                    
        return s_indices, dist_sets

    s_indices, dist_sets = greedy_init()
    best_size = len(s_indices)
    best_s_indices = list(s_indices)
    
    # 3. Simulated Annealing
    current_size = len(s_indices)
    T = 1.0
    cooling = 0.9999
    
    # Helper to check if a candidate index can be added to current state
    def can_add(cand_idx):
        # cand_idx is not in s_indices
        d_vals = []
        for s_idx in s_indices:
            d_vals.append(D[cand_idx, s_idx])
            
        if len(set(d_vals)) != len(d_vals):
            return False
            
        for d2, s_idx in zip(d_vals, s_indices):
            if d2 in dist_sets[s_idx]:
                return False
        return True

    def add_point(cand_idx):
        d_vals = [D[cand_idx, s_idx] for s_idx in s_indices]
        s_indices.append(cand_idx)
        dist_sets[cand_idx] = set(d_vals)
        for d2, s_idx in zip(d_vals, s_indices[:-1]):
            dist_sets[s_idx].add(d2)
            
    def remove_point(idx_to_remove):
        # Find position in s_indices
        pos = s_indices.index(idx_to_remove)
        # Get distances from idx_to_remove to all others
        others = [s for s in s_indices if s != idx_to_remove]
        d_vals = [D[idx_to_remove, s] for s in others]
        
        # Remove from dist_sets of others
        for d2, s_idx in zip(d_vals, others):
            dist_sets[s_idx].remove(d2)
            
        # Remove idx_to_remove from dist_sets
        del dist_sets[idx_to_remove]
        
        # Remove from s_indices
        s_indices.pop(pos)

    # Main SA Loop
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Determine move type
        r = random.random()
        
        if r < 0.7:
            # Try Add
            # Pick random candidate from outside
            # To avoid picking from inside, we can pick random index and check if in s_indices
            # But checking "in" is O(N).
            # Better: maintain a set of s_indices for O(1) check?
            # s_indices is a list. Converting to set every time is O(N).
            # Let's maintain s_set.
            
            # We need to update s_set in add/remove.
            # Let's refactor to use s_set.
            
            pass # Placeholder for logic below
            
        # Let's implement the loop properly with s_set
        
    # Refactor for efficiency
    s_set = set(s_indices)
    
    def try_add(cand_idx):
        if cand_idx in s_set:
            return False
        # Check validity
        valid = True
        d_vals = []
        for s_idx in s_indices:
            d2 = D[cand_idx, s_idx]
            d_vals.append(d2)
            if d2 in dist_sets[s_idx]:
                valid = False
                break
        
        if not valid:
            return False
            
        if len(set(d_vals)) != len(d_vals):
            return False
            
        # Add
        s_indices.append(cand_idx)
        s_set.add(cand_idx)
        dist_sets[cand_idx] = set(d_vals)
        for d2, s_idx in zip(d_vals, s_indices[:-1]):
            dist_sets[s_idx].add(d2)
        return True

    def try_remove(idx_to_remove):
        if idx_to_remove not in s_set:
            return False
        
        others = [s for s in s_indices if s != idx_to_remove]
        d_vals = [D[idx_to_remove, s] for s in others]
        
        for d2, s_idx in zip(d_vals, others):
            dist_sets[s_idx].remove(d2)
            
        del dist_sets[idx_to_remove]
        
        s_indices.remove(idx_to_remove)
        s_set.remove(idx_to_remove)
        return True

    # SA Loop
    no_improve_steps = 0
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.8:
            # Try Add
            cand_idx = random.randint(0, GRID_SIZE - 1)
            if try_add(cand_idx):
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_s_indices = list(s_indices)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # Try Remove (to allow future adds)
            # Pick random from s_indices
            if s_indices:
                idx_to_remove = random.choice(s_indices)
                if try_remove(idx_to_remove):
                    current_size -= 1
                    no_improve_steps += 1
                    # Sometimes accept size reduction to escape local optima
                    # We always accept removals in this simple scheme to maintain validity,
                    # but we should be careful not to shrink too much.
                    # Maybe only remove if current_size > some threshold?
                    # Or use temperature?
                    # Let's add a probability to remove.
                    # If we remove, we lose 1 point.
                    # We should only do this if we are stuck.
                    
        # Temperature update
        T *= cooling
        
        # If stuck for too long, restart or kick
        if no_improve_steps > 1000:
            # Kick: Remove a few random points
            for _ in range(5):
                if s_indices:
                    idx_to_remove = random.choice(s_indices)
                    try_remove(idx_to_remove)
            no_improve_steps = 0
            T = 1.0 # Reset temperature?

    # Convert best_s_indices back to points
    return [points[i] for i in best_s_indices]
