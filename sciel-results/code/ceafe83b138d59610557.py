# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that 
    their maximum squared norm is smaller than or equal to their minimum pairwise squared distance.
    
    Strategy:
    1. Try to find a large set of points with squared norm 3 using a local search heuristic.
    2. If the size is > 220, return that set.
    3. Otherwise, return the set of all 220 points with squared norm 2.
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin
    
    d = 11
    
    # --- Attempt R^2 = 3 ---
    # Generate all vectors with squared norm 3
    candidates = []
    indices = range(d)
    
    for i in indices:
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        for sk in [-1, 1]:
                            p = [0] * d
                            p[i] = si
                            p[j] = sj
                            p[k] = sk
                            candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Check time
    if time.time() > deadline - 30:
        return _get_norm2_points()

    # Precompute dot products
    try:
        dot_prods = candidates @ candidates.T
        # Conflict if dot >= 2
        conflict_mask = dot_prods >= 2
        
        # Local Search for Max Independent Set
        # Start with a greedy solution
        best_set = []
        best_size = 0
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            # Random greedy initialization
            order = np.random.permutation(n_cand)
            available = np.ones(n_cand, dtype=bool)
            selected = []
            
            for idx in order:
                if available[idx]:
                    selected.append(idx)
                    available[idx] = False
                    neighbors = np.where(conflict_mask[idx])[0]
                    available[neighbors] = False
            
            current_size = len(selected)
            selected_set = set(selected)
            
            # Local improvement
            improved = True
            while improved:
                improved = False
                # Try to add nodes
                for idx in range(n_cand):
                    if idx in selected_set:
                        continue
                    # Check if idx conflicts with any selected
                    # conflicts = np.where(conflict_mask[idx, list(selected_set)])[0] # Slow
                    # Better: check against the set
                    # We can maintain a count of conflicts for each node
                    pass # Simplified: just try random additions/swaps if time permits
            
            # A simpler local search: 
            # 1. Try to add any node that has 0 conflicts with current set.
            # 2. If none, try to remove a node and see if we can add more.
            
            # Efficient local search using conflict counts
            # Initialize conflict counts for the selected set
            conflict_counts = np.zeros(n_cand, dtype=int)
            for idx in selected:
                conflict_counts[conflict_mask[idx]] += 1
            # Note: conflict_mask[idx] includes idx itself (dot 3 >= 2).
            # So conflict_counts[idx] >= 1 for selected idx.
            
            selected_bool = np.zeros(n_cand, dtype=bool)
            for idx in selected:
                selected_bool[idx] = True
                
            improved = True
            while improved:
                improved = False
                # Find a node not in set with 0 conflicts
                # conflict_counts[j] == 0 means no selected node conflicts with j
                # But we must ensure j is not in set.
                # For j not in set, if conflict_counts[j] == 0, we can add it.
                
                candidates_to_add = np.where((conflict_counts == 0) & (~selected_bool))[0]
                
                if len(candidates_to_add) > 0:
                    # Add the first one (or random)
                    idx_to_add = candidates_to_add[0]
                    selected_bool[idx_to_add] = True
                    selected.append(idx_to_add)
                    
                    # Update conflict counts
                    # Adding idx_to_add increases conflict count for its neighbors
                    neighbors = np.where(conflict_mask[idx_to_add])[0]
                    conflict_counts[neighbors] += 1
                    improved = True
                else:
                    # No node can be added directly.
                    # Try a swap: remove one node, see if we can add more than 1.
                    # This is complex. Let's just break and try a new random start.
                    break
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected.copy()
                
    except Exception as e:
        return _get_norm2_points()

    if best_size > 220:
        points = candidates[best_set]
        # Verify
        norms_sq = np.sum(points**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = float('inf')
        n_points = len(points)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        if min_dist_sq >= max_norm_sq:
            return points

    # Fallback to R^2 = 2
    return _get_norm2_points()

def _get_norm2_points():
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
