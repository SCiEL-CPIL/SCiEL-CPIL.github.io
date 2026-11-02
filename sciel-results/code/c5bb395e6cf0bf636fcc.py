"""Program for kissing number in dimension 11 using graph heuristic on integer vectors."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    # Type A: One entry +/-2, rest 0.
    # There are 2*D = 22 such vectors.
    vectors = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    vecs_arr = np.array(vectors, dtype=np.int64)
    
    # Type B: Four entries +/-1, rest 0.
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280.
    type_b_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            type_b_vectors.append(v)
            
    tb_arr = np.array(type_b_vectors, dtype=np.int64)
    N_b = len(tb_arr)
    
    # Check compatibility with Type A
    # We established that all Type B are compatible with all Type A.
    # Dot product of Type B (weight 4, +/-1) and Type A (weight 1, +/-2) is +/-2 or 0.
    # Condition: dot <= 2. Always true.
    
    # Now find large Independent Set in Type B conflict graph.
    # Conflict if dot > 2.
    
    # Compute pairwise dot products for Type B
    # tb_arr is (5280, 11)
    # dot_products = tb_arr @ tb_arr.T  # (5280, 5280)
    
    # This creates a large matrix. 5280*5280 ~ 28M floats. ~224MB.
    # To be safe on memory, we can process in chunks or use sparsity?
    # But 224MB is usually acceptable in modern environments.
    # Let's try to be efficient.
    
    # If memory is a concern, we can compute conflicts row by row, but we need the full graph for the greedy step?
    # Actually, for Minimum Degree Greedy, we need to know the neighbors of the selected node to remove them.
    # We can compute the conflict matrix once.
    
    try:
        dots = tb_arr @ tb_arr.T
        conflicts = dots > 2
    except MemoryError:
        # Fallback to greedy if memory fails (unlikely)
        conflicts = None

    if conflicts is not None:
        # Minimum Degree Greedy
        available = np.ones(N_b, dtype=bool)
        selected_indices = []
        
        # Pre-compute degrees? No, degrees change as we remove nodes.
        # We need dynamic degree calculation.
        # To speed up, we can maintain a list of active nodes.
        
        # Simpler: Just iterate.
        # N_b = 5280.
        # In each step, we find min degree among available.
        # Then remove it and its neighbors.
        
        # Optimization: 
        # conflicts is a boolean matrix.
        # When we select i, we must set available[conflicts[i]] = False.
        
        while np.any(available):
            # Calculate degrees of available nodes
            # degree[i] = sum(conflicts[i, available])
            # We can compute this as:
            # degs = conflicts[:, available].sum(axis=1)
            # But this is O(N^2) per step. N=5280. 28M ops per step.
            # If K is 1000, that's 28 Billion ops. Too slow.
            
            # We need a faster way.
            # Alternative: Randomized Greedy?
            # Or, since the graph is not dense, we can just pick a random available node?
            # Random greedy usually gives O(N log N / log log N) for random graphs, but this is structured.
            
            # Let's try a simple heuristic:
            # 1. Shuffle available indices.
            # 2. Iterate through shuffled list.
            # 3. If node is still available, select it, and remove its neighbors.
            
            # This is O(N * (N + K)) roughly?
            # Let's implement the "Shuffle and Sweep" greedy.
            
            # Actually, let's just use the simple greedy from the seed but with a better order?
            # The seed used the natural order.
            # Let's try **Random Order** multiple times and keep the best?
            # Or **Degree-Based** but approximated?
            
            # Let's go with a **Local Search** on the set of selected vectors.
            # Start with a random subset or the seed's result.
            # Try to add random vectors, remove conflicting ones.
            
            # Given the time limit (120s), and N=5280, a simple randomized greedy is likely fast enough.
            # Let's do 1000 random restarts of a simple greedy.
            
            best_set = []
            best_size = 0
            
            # To make it faster, we can just do one pass with a smart order.
            # Order by "popularity" (degree in the full graph).
            # High degree nodes conflict with many, so pick Low degree nodes first.
            
            full_degrees = conflicts.sum(axis=1)
            # Sort indices by full_degree ascending
            order = np.argsort(full_degrees)
            
            available = np.ones(N_b, dtype=bool)
            selected_indices = []
            
            for i in order:
                if not available[i]:
                    continue
                # Select i
                selected_indices.append(i)
                # Remove i and its neighbors
                available[i] = False
                # neighbors are conflicts[i]
                # We need to set available[neighbors] = False
                # conflicts[i] is a boolean vector of length N_b
                available[conflicts[i]] = False
                # Note: This updates available for all neighbors.
                
            # This loop is O(N * (N + K)) if we update the whole vector each time?
            # `available[conflicts[i]] = False` is a vectorized operation, very fast.
            # The loop runs N times.
            # So total time is O(N) vector operations.
            # This is very fast.
            
            # Let's verify the logic.
            # We pick the lowest degree node first.
            # Then we remove it and all its neighbors.
            # This is the standard "Minimum Degree" greedy for Max Independent Set.
            
            selected_vecs = tb_arr[selected_indices]
            
            # Combine with Type A
            all_vecs = np.vstack([vecs_arr, selected_vecs])
            
            # Verify constraints just in case
            # Min pairwise dist >= Max norm
            # Max norm is 2 (norm sq 4).
            # Min dist sq >= 4.
            # dist^2 = 4 + 4 - 2 dot = 8 - 2 dot.
            # 8 - 2 dot >= 4 => 4 >= 2 dot => dot <= 2.
            
            # We ensured dot <= 2 for Type B pairs.
            # Type A and Type B are always compatible.
            # Type A pairs are compatible (dot 0 or -4, both <= 2).
            
            return all_vecs
    else:
        # Fallback to seed logic if memory error
        current_vecs = vecs_arr
        for v in tb_arr:
            dots = current_vecs @ v
            if np.all(dots <= 2):
                current_vecs = np.vstack([current_vecs, v.reshape(1, -1)])
        return current_vecs

# EVOLVE-BLOCK-END
