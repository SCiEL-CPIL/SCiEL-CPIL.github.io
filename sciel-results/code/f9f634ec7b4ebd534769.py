"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

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
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate the trees
    trees = instance_trees(20, 611)
    
    # We want to embed T_1..T_20.
    # Strategy: Greedy with backtracking/random restarts.
    # Order of embedding matters. Smaller trees are easier to pack.
    # Let's try embedding in order 1, 2, ..., 20.
    
    # The graph is K_20. Vertices 0..19.
    # Total edges in K_20 is 190.
    # Sum of edges in trees: T_k has k-1 edges. Sum_{k=1}^{20} (k-1) = 190.
    # So we are trying to decompose K_20 into these 20 trees.
    # This is a tree decomposition problem.
    
    # A simple heuristic: 
    # 1. T_1 is empty.
    # 2. T_2 is an edge.
    # 3. ...
    # 20. T_20 is a spanning tree.
    
    # Since the total edges exactly match, a perfect packing would score 190.
    # However, finding a decomposition is hard. We will use a randomized greedy approach.
    
    # State:
    # used_edges: set of tuples (u, v) with u < v
    # embeddings: list of 20 entries. None or list of 20 ints.
    
    def try_pack(trees, order):
        used_edges = set()
        embeddings = [None] * 20
        
        # Map from tree index in 'trees' list to position in 'embeddings'
        # 'order' is a list of indices into 'trees' representing the embedding sequence
        
        for tree_idx in order:
            if time.time() > deadline:
                return None, embeddings, False
            
            tree = trees[tree_idx]
            n_vertices = tree_idx + 1
            
            # If tree is empty (T_1), map is trivial (any permutation, e.g., identity)
            if n_vertices == 1:
                embeddings[tree_idx] = list(range(20)) # Only vertex 0 is mapped, but we need a list of 20?
                # Contract says: "list of k distinct int host vertices... position i is image of vertex i"
                # So for T_1 (k=1), list has 1 element.
                # Wait, "Entry k-1 is None or a list of k distinct int host vertices".
                # So T_1 -> list of 1 int. T_2 -> list of 2 ints.
                embeddings[tree_idx] = [0]
                continue
            
            # Backtracking search for mapping of vertices 0..n_vertices-1 to 0..19
            # We can use a simple recursive backtracking with randomization.
            
            # To make it faster, we can try to map the tree structure directly.
            # But since K_20 is dense, most mappings work unless edges clash.
            
            # Let's use a randomized greedy with limited backtracking.
            # Map vertices one by one.
            
            # We need to assign host vertices h[0]...h[n_vertices-1]
            # such that for all (u, v) in tree, (h[u], h[v]) not in used_edges.
            
            # We'll do a DFS with a depth limit or node limit to avoid exponential blowup.
            # Since n is small (max 20), and K_20 is dense, it should be fast.
            
            # Optimization: 
            # We can fix the mapping of vertex 0 to a random vertex to break symmetry? 
            # No, we need distinct hosts.
            
            # Let's use a simple backtracking solver.
            
            def solve_tree(tree, n_v):
                h = [None] * n_v
                # Try to assign h[0] to some random vertex? 
                # Actually, we can just iterate.
                
                # To speed up, we can order the tree vertices by degree or just index.
                # Let's just index 0..n_v-1.
                
                # We need to ensure h[u] != h[v] for all u!=v.
                
                # We'll use a recursive function.
                # To avoid deep recursion issues, we use a stack or limit depth.
                # But n_v is at most 20.
                
                # Let's try a randomized greedy first, then backtrack if stuck.
                
                # Actually, a full backtracking is safest for correctness.
                # With 20 vertices, worst case is 20! which is too big.
                # But constraints are strong (edges must be disjoint).
                
                # Better approach:
                # We know the tree structure.
                # We can map the tree into the host graph K_20.
                # Since K_20 is complete, the only constraint is edge disjointness with previous trees.
                
                # Let's use a simple DFS.
                # We will pick a random order for vertices? No, tree structure dictates dependencies.
                # We can root the tree at 0.
                # Parent of i is determined by the tree edges.
                
                # Build adjacency list for the tree
                adj = [[] for _ in range(n_v)]
                for u, v in tree:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Root at 0
                parent = [-1] * n_v
                order = [0]
                parent[0] = -2 # Mark root
                for i in range(n_v):
                    if i == 0: continue
                    # Find parent in BFS/DFS
                    pass # We need to build the tree structure properly
                
                # Let's just build a list of (child, parent) pairs for traversal
                # We need to map vertices in an order such that parent is mapped before child.
                # BFS from 0.
                
                visited = [False] * n_v
                visited[0] = True
                queue = [0]
                map_order = [0]
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                            map_order.append(v)
                
                # Now map_order is a valid order for assignment.
                # We will assign h[v] for v in map_order.
                
                # We need to choose h[v] from available vertices 0..19.
                # Available means:
                # 1. Not used by any other h[u] for u already assigned.
                # 2. For every neighbor p of v that is already assigned (i.e., parent), 
                #    the edge (h[p], h[v]) is not in used_edges.
                #    Note: children are not assigned yet, so we only check parents.
                
                # This is a constraint satisfaction problem.
                
                # We will use backtracking on map_order.
                
                def backtrack(idx):
                    if time.time() > deadline:
                        return False
                    if idx == n_v:
                        return True
                    
                    v = map_order[idx]
                    # Get available candidates for v
                    candidates = []
                    for hv in range(20):
                        # Check if hv is already used
                        if hv in h:
                            continue
                        
                        # Check edges to already assigned neighbors
                        ok = True
                        for p in adj[v]:
                            if h[p] is not None:
                                e1, e2 = sorted((h[p], hv))
                                if (e1, e2) in used_edges:
                                    ok = False
                                    break
                        if ok:
                            candidates.append(hv)
                    
                    # Randomize candidates to help escape local minima / find diverse solutions
                    random.shuffle(candidates)
                    
                    for hv in candidates:
                        h[v] = hv
                        if backtrack(idx + 1):
                            return True
                        h[v] = None
                    
                    return False
                
                # To improve success rate, we can try multiple random shuffles of candidates?
                # Or just one deep search.
                # Since K_20 is dense, solutions should be abundant.
                
                if backtrack(0):
                    return h
                return None
            
            mapping = solve_tree(tree, n_vertices)
            if mapping is None:
                return None, embeddings, False
            
            # Record mapping
            embeddings[tree_idx] = mapping
            
            # Update used_edges
            for u, v in tree:
                e1, e2 = sorted((mapping[u], mapping[v]))
                used_edges.add((e1, e2))
        
        return used_edges, embeddings, True

    # We will try a few different orders or random restarts if the first fails.
    # Given the density, the first attempt (1..20) is likely to succeed.
    # But to be safe, we can randomize the order of embedding trees.
    # Smaller trees are easier, so 1..20 is a good heuristic.
    
    # Let's try 1..20 first.
    order = list(range(20))
    res, embeddings, success = try_pack(trees, order)
    
    if success:
        # Verify types
        final_result = []
        for i in range(20):
            if embeddings[i] is None:
                final_result.append(None)
            else:
                # Ensure it's a list of ints
                final_result.append([int(x) for x in embeddings[i]])
        return final_result
    
    # If it failed (unlikely), try a randomized order
    # Or just return the partial result? 
    # The contract says "Any broken rule makes the whole result invalid".
    # So we must return a valid packing or None for all?
    # "Entry k-1 is None or a list...".
    # If we can't pack T_k, we can set it to None.
    # But "Embedded trees must be edge-disjoint".
    # If we skip some, the remaining ones are still edge-disjoint.
    # So we can return a partial packing.
    
    # Let's try to pack as many as possible in order 1..20, stopping on failure.
    used_edges = set()
    final_embeddings = [None] * 20
    
    for tree_idx in range(20):
        if time.time() > deadline:
            break
        tree = trees[tree_idx]
        n_v = tree_idx + 1
        if n_v == 1:
            final_embeddings[tree_idx] = [0]
            continue
            
        # Re-use the solver logic inline or call a helper
        # We'll just call a simplified version that doesn't check global deadline inside backtrack too aggressively
        # but we already have the deadline guard in try_pack.
        
        # Let's just re-run try_pack but allowing partial success?
        # The current try_pack returns None on failure.
        
        # Let's modify the strategy: 
        # Try to embed T_1, then T_2, etc.
        # If one fails, try randomizing the host mapping for that tree with more restarts.
        
        # For simplicity and robustness within the time limit, 
        # I will implement a "greedy with restarts" for each tree.
        
        max_restarts = 50
        success = False
        for _ in range(max_restarts):
            # Randomize the candidate order in the solver
            # We can pass a random seed or just rely on random.shuffle inside
            mapping = solve_tree(tree, n_v)
            if mapping:
                final_embeddings[tree_idx] = mapping
                for u, v in tree:
                    e1, e2 = sorted((mapping[u], mapping[v]))
                    used_edges.add((e1, e2))
                success = True
                break
        
        if not success:
            # If we can't embed this tree, we can skip it (set to None)
            # But subsequent trees might still be embeddable?
            # Yes, because we just don't use the edges.
            # So we continue to the next tree.
            pass

    # Construct final result
    final_result = []
    for i in range(20):
        if final_embeddings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in final_embeddings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
