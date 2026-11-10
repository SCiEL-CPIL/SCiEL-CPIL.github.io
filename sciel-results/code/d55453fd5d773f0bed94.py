import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit
    
    num_points = N * N
    
    # Precompute coordinates for all points
    # Shape: (num_points, 2)
    idxs = np.arange(num_points)
    coords = np.array([(i // N, i % N) for i in idxs], dtype=np.int32)
    
    best_size = 0
    best_selected = []
    
    # We will run multiple randomized greedy passes
    # Each pass:
    # 1. Shuffle all point indices
    # 2. Iterate through them, adding to set if valid
    # 3. Update best if larger set found
    
    # To speed up, we can also try to "repair" or "extend" existing sets, 
    # but simple repeated greedy with random ordering is often very strong for this size.
    
    rng = random.Random(12345)
    
    while time.time() < deadline:
        # 1. Generate a random ordering of points
        order = list(range(num_points))
        rng.shuffle(order)
        
        # 2. Greedy construction
        selected = []
        # selected_coords will be a list of coordinate tuples or we can maintain a NumPy array
        # For speed in 'can_add', we want a NumPy array of selected coords.
        # However, adding to a NumPy array is O(N) due to copy. 
        # Given K is small (~10-20), this is fine.
        # Alternatively, keep a list and convert to array for the check? 
        # Let's keep a list of indices and compute coords on the fly or maintain a small array.
        
        # Let's maintain `selected_indices` and a helper to get coords.
        # Actually, `coords` is a global array. 
        # `selected_coords` can be a list of arrays? No, let's just use a list of indices 
        # and slice `coords` when needed. 
        # `coords[selected_indices]` is a vectorized operation.
        
        selected_indices = []
        
        # We also need to track distances from each selected point to others for conflict checks.
        # dist_sets[i] is a set of squared distances from selected_indices[i] to other selected points.
        dist_sets = []
        
        def _can_add(p_idx):
            if not selected_indices:
                return True
            
            p = coords[p_idx]
            # Get coords of all selected points
            # This creates a view/copy of shape (K, 2)
            sel_coords = coords[selected_indices]
            
            # Compute squared distances
            # diff shape (K, 2)
            diff = sel_coords - p
            dists = np.sum(diff * diff, axis=1)
            
            # Check 1: Uniqueness of distances from p
            # If K is small, np.unique is fast. 
            # Alternatively, check if len(set(dists)) == K
            # np.unique returns sorted array.
            if len(np.unique(dists)) != len(selected_indices):
                return False
            
            # Check 2: Conflicts with existing distance sets
            # For each selected point i, dists[i] must not be in dist_sets[i]
            # dists is a NumPy array. dist_sets is a list of sets.
            # We must loop in Python here.
            for i in range(len(selected_indices)):
                d_val = dists[i]
                if d_val in dist_sets[i]:
                    return False
                    
            return True
            
        def _add(p_idx):
            selected_indices.append(p_idx)
            dist_sets.append(set())
            
            # Update dist_sets for all existing points
            p = coords[p_idx]
            K = len(selected_indices)
            # Get coords of all points including the new one
            # We need to compute distances from p to all previous points
            # and add them to their respective sets.
            
            # It's more efficient to compute distances from p to all *previous* points
            # and add them to the sets.
            
            for i in range(K - 1):
                q = coords[selected_indices[i]]
                d = int(np.sum((p - q) * (p - q)))
                dist_sets[i].add(d)
                dist_sets[K-1].add(d)
        
        # Run the greedy pass
        for p_idx in order:
            if _can_add(p_idx):
                _add(p_idx)
                
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_selected = selected_indices[:]
            
    # Convert best_selected indices back to (x, y) tuples
    result = []
    for idx in best_selected:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
