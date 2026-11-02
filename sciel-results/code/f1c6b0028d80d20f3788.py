import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # Generate all weight 3 vectors
    vectors = []
    indices = range(D)
    # Use itertools for combinations
    import itertools
    
    for combo in itertools.combinations(indices, 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            
    V = len(vectors)
    
    # Build adjacency list for conflicts
    # Conflict if dot product > 1 (i.e., 2 or 3)
    # Since vectors are distinct, dot=3 is impossible.
    # So conflict if dot == 2.
    
    adj = [set() for _ in range(V)]
    
    # Optimization: Group by support?
    # Or just O(V^2) dot product check.
    # 1320^2 = 1.7M operations. Very fast in Python if vectorized or simple loops.
    # Let's use numpy for speed if possible, or just python loops.
    # Python loops for 1.7M iterations might take ~1-2 seconds. Acceptable.
    
    vecs_np = np.array(vectors, dtype=np.int8)
    
    # Calculate dot products
    # We only need to know if dot > 1.
    # dot = vecs_np @ vecs_np.T
    # This creates a 1320x1320 matrix. ~1.7M floats. Memory ~13MB. Fast.
    
    dots = vecs_np @ vecs_np.T
    
    # Create adjacency list
    # i, j are neighbors if dots[i,j] > 1
    # Note: i != j
    
    for i in range(V):
        # Find neighbors
        # np.where(dots[i] > 1)[0]
        # Exclude self
        neighbors = np.where(dots[i] > 1)[0]
        neighbors = neighbors[neighbors != i]
        adj[i] = set(neighbors.tolist())
        
    if time.time() > deadline:
        # Fallback to seed if construction takes too long (unlikely)
        return np.array([[1,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    best_solution = []
    best_size = 0
    
    # Run Greedy MIS multiple times
    # Strategy: Randomly shuffle node order, then pick min degree?
    # Or just standard Min Degree Greedy.
    
    # Let's implement a slightly smarter greedy:
    # 1. Start with all nodes available.
    # 2. Pick node with min degree.
    # 3. Add to solution.
    # 4. Remove node and neighbors.
    
    # To improve, we can try different tie-breaking strategies.
    
    for attempt in range(50): # 50 attempts
        if time.time() > deadline - 10:
            break
            
        available = set(range(V))
        solution = []
        
        # To speed up min-degree finding, we could maintain degrees,
        # but V is small (1320). Recalculating degree is O(deg).
        
        while available:
            # Find node with min degree in available subgraph
            min_deg = float('inf')
            min_node = -1
            for u in available:
                deg = len(adj[u] & available)
                if deg < min_deg:
                    min_deg = deg
                    min_node = u
                    if deg == 0:
                        break
            
            if min_node == -1:
                break
                
            solution.append(min_node)
            # Remove min_node and its neighbors
            to_remove = adj[min_node] | {min_node}
            available -= to_remove
            
        if len(solution) > best_size:
            best_size = len(solution)
            best_solution = solution.copy()
            
    # Convert solution indices to vectors
    if best_solution:
        result_vectors = [vectors[i] for i in best_solution]
        # Try to add weight 2 vectors
        # Generate all weight 2 vectors
        w2_vectors = []
        for combo in itertools.combinations(indices, 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for idx, val in zip(combo, signs):
                    v[idx] = val
                w2_vectors.append(v)
                
        # Check each w2 vector against the current best set
        # A w2 vector u can be added if for all v in best_set, dot(u,v) <= 1
        # (since R^2=3, condition is 5 - 2dot >= 3 => dot <= 1)
        
        added_w2 = []
        for u in w2_vectors:
            u_np = np.array(u, dtype=np.int8)
            # Check against all selected w3 vectors
            # We need to be careful: if we add u, does it conflict with other added w2?
            # W2 vs W2 conflict:
            # R^2 is still 3 (max norm is sqrt(3)).
            # Condition for two w2 vectors u1, u2:
            # ||u1-u2||^2 >= 3.
            # 2 + 2 - 2 dot >= 3 => 4 - 2 dot >= 3 => 2 dot <= 1 => dot <= 0.5 => dot <= 0.
            # So w2 vectors must have dot product <= 0 with each other.
            
            # This makes adding w2 more complex. We need an independent set in the w2 graph
            # that is also compatible with the w3 set.
            
            # Given time constraints, let's just check compatibility with W3 first.
            # If it passes, we add it to a candidate list.
            # Then we solve MIS on the candidate W2 list (with edge if dot > 0).
            
            compatible = True
            for v_idx in best_solution:
                v = vectors[v_idx]
                dot = np.dot(u_np, v)
                if dot > 1:
                    compatible = False
                    break
            
            if compatible:
                added_w2.append(u)
                
        if added_w2:
            # Solve MIS for added_w2
            # Graph: nodes are added_w2, edge if dot > 0
            # Size is at most 220.
            # We can use a simple greedy or even brute force if small.
            # Let's use greedy again.
            
            w2_adj = [set() for _ in range(len(added_w2))]
            for i in range(len(added_w2)):
                for j in range(i+1, len(added_w2)):
                    dot = np.dot(added_w2[i], added_w2[j])
                    if dot > 0:
                        w2_adj[i].add(j)
                        w2_adj[j].add(i)
                        
            # Greedy MIS on w2_adj
            avail_w2 = set(range(len(added_w2)))
            w2_sol = []
            while avail_w2:
                min_deg = float('inf')
                min_node = -1
                for u in avail_w2:
                    deg = len(w2_adj[u] & avail_w2)
                    if deg < min_deg:
                        min_deg = deg
                        min_node = u
                if min_node == -1: break
                w2_sol.append(min_node)
                to_remove = w2_adj[min_node] | {min_node}
                avail_w2 -= to_remove
                
            # Add these to final result
            for idx in w2_sol:
                result_vectors.append(added_w2[idx])
                
        result = np.array(result_vectors, dtype=np.int64)
        return result
    else:
        # Fallback
        return np.array(vectors[:220], dtype=np.int64)
