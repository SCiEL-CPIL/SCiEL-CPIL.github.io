import numpy as np
import time
import itertools
from collections import deque

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin (evaluator kills at 630s)
    
    # 1. Generate all integer vectors with squared norm 2
    # These are vectors with two non-zero entries, each +/-1.
    # Count: C(11,2) * 2^2 = 55 * 4 = 220
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_2.append(v)
    
    base_solution = np.array(candidates_2, dtype=np.int64)
    best_points = base_solution.copy()
    best_k = len(base_solution)
    
    # 2. Generate all integer vectors with squared norm 4
    # These are vectors with four non-zero entries, each +/-1.
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_4.append(v)
    
    if not candidates_4:
        return best_points
        
    V4 = np.array(candidates_4, dtype=np.int64)
    N4 = len(V4)
    
    # 3. Build Conflict Graph for Norm 4 vectors
    # We want to add as many norm-4 vectors as possible to the base solution
    # such that the minimum squared distance between ANY pair of points in the final set is >= max_norm_sq.
    # The base solution has norm sq = 2.
    # If we add a norm-4 vector v, its norm sq is 4.
    # The max_norm_sq of the combined set will be 4.
    # Therefore, we require min_dist_sq >= 4.
    
    # Distance squared between base point u (norm 2) and new point v (norm 4):
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u, v> = 2 + 4 - 2<u, v> = 6 - 2<u, v>
    # We need 6 - 2<u, v> >= 4  =>  2 - 2<u, v> >= 0  =>  <u, v> <= 1.
    
    # Distance squared between two new points v, w (both norm 4):
    # ||v - w||^2 = 4 + 4 - 2<v, w> = 8 - 2<v, w>
    # We need 8 - 2<v, w> >= 4  =>  4 - 2<v, w> >= 0  =>  <v, w> <= 2.
    
    # So a norm-4 vector v is "compatible" with the base set if <u, v> <= 1 for all u in base.
    # And two norm-4 vectors v, w are compatible with each other if <v, w> <= 2.
    
    # Step 3a: Filter V4 vectors that are compatible with the base set.
    # Base set has 220 vectors. V4 has 5280 vectors.
    # Compute dot products: (5280, 11) @ (11, 220) -> (5280, 220).
    # Check if max dot product for each v is <= 1.
    
    dots_base = V4 @ base_solution.T
    max_dots_base = np.max(dots_base, axis=1)
    valid_mask = max_dots_base <= 1
    V4_valid = V4[valid_mask]
    N_valid = len(V4_valid)
    
    if N_valid == 0:
        return best_points
        
    # Step 3b: Build conflict graph among V4_valid vectors.
    # Conflict if dot product > 2.
    dots_valid = V4_valid @ V4_valid.T
    # We only care about the upper triangle for adjacency, but for greedy we need full adjacency or efficient neighbor lookup.
    # N_valid is likely smaller than 5280. Let's estimate.
    # A random v in V4 has 4 non-zeros. A random u in base has 2 non-zeros.
    # Probability of overlap is roughly small.
    # Actually, let's just compute the adjacency list for the valid set.
    
    # To save memory and time, we can use a bitset or just lists of neighbors.
    # N_valid is at most 5280.
    # Let's build adjacency lists.
    
    # First, identify conflicts.
    # conflict[i, j] is True if dots_valid[i, j] > 2.
    # Note: dots_valid[i, i] = 4, which is > 2, so self-conflict exists. We ignore self.
    
    conflict_adj = [None] * N_valid
    for i in range(N_valid):
        if time.time() > deadline - 20:
            # Return best found so far (which is just base + maybe some from previous iterations if we had them, but here it's just base)
            # Wait, we haven't added any V4 yet.
            return best_points
        
        # Find neighbors j != i such that dot > 2
        neighbors = np.where((dots_valid[i] > 2) & (np.arange(N_valid) != i))[0]
        conflict_adj[i] = neighbors

    del dots_valid

    # Step 4: Find Max Independent Set (MIS) in the conflict graph of V4_valid.
    # We use a greedy algorithm with random restarts.
    # Since the graph is likely sparse (conflicts are local), greedy might work well.
    
    best_indices = []
    
    # Optimization: Precompute degrees to sort by degree?
    # In greedy MIS, picking low-degree nodes first often yields better results.
    # Let's compute degrees.
    degrees = np.array([len(adj) for adj in conflict_adj], dtype=np.int32)
    
    while time.time() < deadline - 10:
        # Random restart
        # Shuffle order
        order = np.random.permutation(N_valid)
        
        # active_mask: True if vertex is still available
        active_mask = np.ones(N_valid, dtype=bool)
        current_indices = []
        
        # To speed up, we can iterate through the shuffled order.
        for v in order:
            if not active_mask[v]:
                continue
            current_indices.append(v)
            # Deactivate v and its neighbors
            active_mask[v] = False
            if len(conflict_adj[v]) > 0:
                active_mask[conflict_adj[v]] = False
                
        if len(current_indices) > len(best_indices):
            best_indices = current_indices
            
            # Update best_points
            # Combine base_solution and V4_valid[best_indices]
            new_points = np.vstack([base_solution, V4_valid[best_indices].astype(np.int64)])
            best_points = new_points
            best_k = len(new_points)
            
            # Check if we should break early? No, keep searching.
            
    return best_points

if __name__ == "__main__":
    # Quick self-test
    points = kissing_number11()
    print(f"Found {len(points)} points")
    # Verify
    if len(points) > 0:
        norms = np.sum(points**2, axis=1)
        max_norm = np.max(norms)
        min_dist_sq = float('inf')
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        print(f"Max norm sq: {max_norm}, Min dist sq: {min_dist_sq}")
        if min_dist_sq >= max_norm:
            print("Valid!")
        else:
            print("INVALID!")
