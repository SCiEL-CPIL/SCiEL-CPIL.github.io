import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe (kill at 150, buffer 50)
    
    # 1. Generate Weight-4 vectors
    v4_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            v4_list.append(v)
    V4 = np.array(v4_list, dtype=np.int8)
    
    # 2. Generate satellite vectors: 2*e_i for each coordinate i
    #    These have norm 2, and dot with any weight-4 vector is in {-2, 0, 2}
    #    So dot <= 2 always holds (compatible!)
    #    Also dot between two satellites: 2*e_i . 2*e_j = 4 if i==j, 0 if i!=j
    #    So satellites conflict with themselves but not with others.
    #    We can include at most one satellite per coordinate, but since they're
    #    mutually compatible (dot=0 for i!=j), we can include ALL 22!
    satellite_list = []
    for i in range(D):
        v = [0] * D
        v[i] = 2
        satellite_list.append(v)
        v2 = [0] * D
        v2[i] = -2
        satellite_list.append(v2)
    V_sat = np.array(satellite_list, dtype=np.int8)
    
    # Combine: core (weight-4) + satellites
    V = np.vstack([V4, V_sat])
    n_vecs = V.shape[0]
    n_core = V4.shape[0]
    n_sat = V_sat.shape[0]
    
    # Compute dot products
    dots = V @ V.T
    
    # Conflict if dot > 2
    # We want MIS in graph where edge exists if dots[i,j] > 2.
    # Note: diagonal is 4, so self is conflict.
    
    best_size = 0
    best_indices = []
    
    # Pre-compute conflict masks as bool array
    # conflict_masks[i, j] is True if i and j conflict
    conflict_masks = (dots > 2).astype(np.bool_)
    
    # Minimum Degree Greedy with random restarts
    # We use a fixed number of restarts or time-based
    num_restarts = 200
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomize tie-breaking by permuting nodes? 
        # Or just use argmin which picks first min. 
        # To get diversity, we can add small random noise to degrees?
        # Or shuffle the order of nodes in the initial available set?
        # The min-degree heuristic is deterministic if degrees are equal.
        # Let's add small random noise to degrees to break ties randomly.
        
        available = np.ones(n_vecs, dtype=bool)
        selected = []
        
        while True:
            if time.time() > deadline:
                break
            
            avail_indices = np.nonzero(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Calculate degrees within available subgraph
            # Submatrix: conflict_masks[avail_indices][:, avail_indices]
            # We want row sums of this submatrix.
            # Since conflict_masks is bool, we can sum.
            # This creates a view/copy of size M x M where M is current available count.
            # M starts at 3360, decreases.
            # Summing MxM matrix is O(M^2).
            # Total cost is sum of M^2 over steps.
            # This might be too slow if M stays large for many steps.
            
            # Optimization:
            # Instead of full submatrix sum, we can maintain a degree array and update it.
            # But updating degrees for all neighbors of u is O(degree(u)).
            # Total cost O(E) per restart. E ~ 3360 * 64 / 2 ~ 100k.
            # This is much faster than O(N^2) per step.
            
            # Let's implement dynamic degree update.
            # 1. Init degrees = conflict_masks.sum(axis=1).astype(np.int32)
            # 2. available = ones
            # 3. Loop:
            #    a. Find u in available with min degree.
            #    b. Select u.
            #    c. For each v in neighbors(u) U {u}:
            #         if available[v]:
            #             available[v] = False
            #             for each w in neighbors(v) U {v} (that are still available? No, just update degree count for available nodes?
            #             Actually, we just need to decrement degree of w for each removed neighbor.
            #             This is tricky to do efficiently with NumPy without loops.
            
            # Given N=3360, the simple "sum submatrix" approach might be okay for the first few steps, but slows down.
            # Let's try the simple approach first and see if it fits in time.
            # If M=3360, M^2 = 11M. 
            # If we do this 100 times, 1.1B ops. In NumPy, ~1-2s?
            # If we do this 1000 times (solution size), 11B ops. Too slow.
            
            # So we MUST use the dynamic update or a smarter heuristic.
            # Let's try a different heuristic: **Random Greedy with Degree Bias**.
            # Pick node with probability proportional to 1/(degree+1).
            # This is fast: 
            # 1. Compute degrees (vector).
            # 2. Weights = 1.0 / (degrees[available] + 1).
            # 3. Normalize weights.
            # 4. Sample u.
            # 5. Update available.
            # 6. Recalculate degrees? No, we can just recompute degrees from scratch each step?
            #    Recomputing degrees: `conflict_masks @ available`. O(N^2) per step.
            #    Same problem.
            
            # Let's go back to **Tomita's Max Clique** on the complement graph.
            # The complement graph has edges where dot <= 2.
            # This graph is DENSE.
            # Max Clique in dense graphs is fast with Tomita.
            # But implementing Tomita in Python is complex.
            
            # Alternative: **Coloring based pruning** is part of Tomita.
            
            # Let's try a simpler structural change: **Greedy with Random Restart and "Low Degree" tie-breaking**.
            # We will pre-compute `neighbors` as a list of lists for fast iteration.
            # `neighbors[i]` is list of j such that conflict_masks[i,j] is True.
            
            # Dynamic Degree Update Implementation:
            # `deg[i]` = number of available neighbors of i (including self if available? No, self is not a neighbor in MIS context usually, but here self is conflict. So if i is available, it "blocks" itself. We remove i and its neighbors. So effectively, we pick u, then remove u and N(u).
            # So we don't need to track "self" in degree for selection, we just pick u from available.
            
            # Let's use the list-of-neighbors approach.
            
            if attempt == 0:
                # Build neighbor lists
                neighbors = []
                for i in range(n_vecs):
                    nbrs = np.nonzero(conflict_masks[i])[0]
                    neighbors.append(nbrs)
            
            # Init degrees
            # degree[i] = number of available neighbors (excluding self? 
            # If we pick u, we remove u and all v in neighbors[u].
            # So the "cost" of picking u is 1 + len(neighbors[u] intersect available).
            # We want to minimize this cost.
            
            available = np.ones(n_vecs, dtype=bool)
            # deg[i] is count of available nodes in neighbors[i] (which includes i? 
            # conflict_masks[i,i] is True. So neighbors[i] includes i.
            # So if we pick u, we remove u and all v in neighbors[u].
            # The number of removed nodes is len(neighbors[u] intersect available).
            # We want to minimize this.
            
            # Initialize deg
            deg = np.array([len(n) for n in neighbors], dtype=np.int32)
            # Note: deg[i] counts all neighbors, including self.
            
            selected = []
            
            while True:
                if time.time() > deadline:
                    break
                
                # Find available nodes
                avail_indices = np.nonzero(available)[0]
                if len(avail_indices) == 0:
                    break
                
                # Find min degree among available
                # We need deg[avail_indices]
                # Add random noise to break ties?
                # noise = np.random.rand(len(avail_indices)) * 0.1
                # costs = deg[avail_indices].astype(float) + noise
                # min_idx = np.argmin(costs)
                
                # To be deterministic but diverse, let's just pick min. 
                # If ties, argmin picks first. 
                # To randomize, we can shuffle avail_indices? No, that changes which one is "first".
                # Let's add small noise.
                noise = np.random.rand(len(avail_indices)) * 1e-6
                costs = deg[avail_indices].astype(float) + noise
                min_local_idx = np.argmin(costs)
                u = avail_indices[min_local_idx]
                
                selected.append(u)
                
                # Remove u and its neighbors
                # removed = neighbors[u]
                # We need to update deg for all remaining available nodes.
                # For each v in removed:
                #   if available[v]:
                #      available[v] = False
                #      For each w in neighbors[v]:
                #         if available[w]:
                #            deg[w] -= 1
            
                removed = neighbors[u]
                # Iterate through removed nodes
                for v in removed:
                    if available[v]:
                        available[v] = False
                        # Update degrees of v's neighbors
                        # deg[w] -= 1 for w in neighbors[v] if available[w]
                        # This inner loop is expensive in Python.
                        # neighbors[v] is a NumPy array.
                        # We can vectorize:
                        # v_nbrs = neighbors[v]
                        # mask = available[v_nbrs]
                        # deg[v_nbrs[mask]] -= 1
                        # This is still a loop over v in removed.
                        # len(removed) ~ 65.
                        # So 65 iterations per step.
                        # Total steps ~ 3000.
                        # 3000 * 65 = 195k iterations.
                        # Each iteration does NumPy ops on array of size ~65.
                        # 195k * 65 ~ 12M ops.
                        # This is very fast!
                        
                        v_nbrs = neighbors[v]
                        mask = available[v_nbrs]
                        if np.any(mask):
                            deg[v_nbrs[mask]] -= 1
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_indices = selected
                if best_size > 3500:
                    break

    if best_size == 0:
        return V.astype(np.int64)

    return V[best_indices].astype(np.int64)
