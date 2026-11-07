# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 115.0  # safety margin
    n = 20
    trees = instance_trees(n, 611)
    
    # Track used edges
    used_edges = set()
    result = [None] * n
    
    # We will embed trees from largest to smallest.
    # T20 has 19 edges, T19 has 18, etc.
    # This order is generally better for packing because larger structures are harder to fit.
    order = list(range(n - 1, -1, -1))
    
    for idx in order:
        if time.time() > deadline:
            break
        k = idx + 1
        tree_edges = trees[idx]
        
        if k == 1:
            # Single vertex, any unused vertex works? 
            # Wait, "embedded trees must be edge-disjoint". 
            # A single vertex has no edges. So it doesn't consume any edges.
            # But the host vertices must be distinct? 
            # "list of k distinct int host vertices".
            # So T1 just needs 1 distinct vertex.
            # Since no edges are used, it doesn't conflict with edge-disjointness.
            # But does it conflict with vertex-disjointness? 
            # The prompt says "embedded trees must be edge-disjoint in K20".
            # It does NOT say vertex-disjoint.
            # So vertices can be shared.
            # However, for k=1, we just need to pick a vertex.
            # Let's pick 0.
            result[idx] = [0]
            continue
        
        if k == 2:
            # Single edge (0,1).
            # Find any unused edge.
            found = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if (u, v) not in used_edges and (v, u) not in used_edges:
                        result[idx] = [u, v]
                        used_edges.add((u, v))
                        used_edges.add((v, u))
                        found = True
                        break
                if found:
                    break
            if not found:
                # Should not happen unless we packed 190 edges already
                pass
            continue
        
        # For k >= 3, use backtracking
        # Build adjacency list for the tree
        tree_adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            tree_adj[a].append(b)
            tree_adj[b].append(a)
        
        # Backtracking search
        # State: mapping of tree vertex -> host vertex
        # We will map tree vertices one by one.
        # Heuristic: Start with a high-degree vertex in the tree.
        # Then pick the next tree vertex to assign based on the number of available host neighbors.
        
        # Find a starting vertex (max degree in tree)
        start_v = max(range(k), key=lambda v: len(tree_adj[v]))
        
        # Try to embed
        embedding = [-1] * k
        embedding[start_v] = 0 # Start with host 0, but we should try all host vertices
        
        # To be robust, we can iterate over possible host vertices for the start node
        # But backtracking will handle it.
        
        def backtrack(assigned_count):
            if time.time() > deadline:
                return False
            
            if assigned_count == k:
                # Verify all edges are disjoint (should be true if we checked during assignment)
                return True
            
            # Select the next tree vertex to assign
            # Heuristic: Among unassigned vertices, pick the one with the most constraints
            # i.e., the one with the most neighbors already assigned.
            best_v = -1
            best_score = -1
            for v in range(k):
                if embedding[v] != -1:
                    continue
                # Count assigned neighbors
                assigned_neighbors = sum(1 for u in tree_adj[v] if embedding[u] != -1)
                # Prefer vertices with more assigned neighbors (more constrained)
                if assigned_neighbors > best_score:
                    best_score = assigned_neighbors
                    best_v = v
            
            if best_v == -1:
                # No unassigned vertex? Should not happen if assigned_count < k
                return False
                
            # Try all possible host vertices for best_v
            # To improve speed, we can shuffle or order them
            # Host vertices that have more available connections to assigned neighbors are better?
            # Actually, we just need to check if the edge to the assigned neighbor is free.
            
            # Gather candidate host vertices
            candidates = list(range(n))
            random.shuffle(candidates) # Randomization to avoid getting stuck in local minima
            
            for h in candidates:
                # Check if h is valid for best_v
                # 1. Injectivity: h must not be used by another vertex in this tree
                if h in embedding:
                    continue
                
                # 2. If best_v has an assigned neighbor, the edge (embedding[neighbor], h) must be free.
                # 3. If best_v has multiple assigned neighbors, all those edges must be free.
                valid = True
                for u in tree_adj[best_v]:
                    if embedding[u] != -1:
                        u_host = embedding[u]
                        if (u_host, h) in used_edges or (h, u_host) in used_edges:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign h to best_v
                embedding[best_v] = h
                
                # Recurse
                if backtrack(assigned_count + 1):
                    return True
                
                # Unassign
                embedding[best_v] = -1
                
            return False
        
        # We need to try different starting host vertices or just let backtracking handle it?
        # The current backtracking assumes embedding[start_v] is set.
        # Let's modify: start_v is the first to be assigned.
        # So we loop over possible host vertices for start_v.
        
        found = False
        for h in range(n):
            if time.time() > deadline:
                break
            embedding = [-1] * k
            embedding[start_v] = h
            if backtrack(1):
                # Success
                result[idx] = embedding[:]
                # Add edges to used set
                for a, b in tree_edges:
                    ua, ub = embedding[a], embedding[b]
                    used_edges.add((ua, ub))
                    used_edges.add((ub, ua))
                found = True
                break
        
        if not found:
            # If backtracking fails, maybe try a different start vertex?
            # Or just leave it as None.
            pass

    # Convert to proper format
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append(list(result[i]))
    
    return final_result
# EVOLVE-BLOCK-END
