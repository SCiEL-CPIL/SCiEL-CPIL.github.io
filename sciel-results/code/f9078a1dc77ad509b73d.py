"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
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
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate the fixed trees
    trees = instance_trees(20, 611)
    
    # Pre-compute adjacency lists for each tree to speed up neighbor checks
    tree_adj = []
    for edges in trees:
        adj = [[] for _ in range(len(edges) + 1 if edges else 1)] # k vertices
        k = len(edges) + 1 if edges else 1
        if k == 1:
            tree_adj.append([])
            continue
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        
    # K20 has 190 edges. Each tree T_k has k-1 edges.
    # Total edges to pack: sum(k-1 for k in 1..20) = 190.
    # This means we are looking for a decomposition of K20 into these specific trees.
    # This is a very tight constraint. If a perfect packing exists, it uses all 190 edges.
    
    # State:
    # used_edges: set of tuples (min(u,v), max(u,v))
    # mapping: list of 20 entries. entry k-1 is None or list of k ints.
    
    # We will use backtracking.
    # Order of embedding: Smallest trees first? Largest first?
    # Smallest trees have fewer constraints on vertex mapping (fewer vertices to place), 
    # but they use specific edges.
    # Actually, let's try embedding in order of increasing size (1 to 20).
    # T1: 0 edges. Trivial.
    # T2: 1 edge.
    # ...
    # T20: 19 edges.
    
    # However, T20 is a spanning tree. It must use 19 edges.
    # If we embed small trees first, we might block the large tree.
    # If we embed T20 first, we use 19 edges. Then we need to pack T1..T19 into the remaining 171 edges.
    # The remaining graph is K20 minus a spanning tree.
    
    # Let's try a randomized backtracking approach.
    # We will attempt to find a valid packing.
    # Since the total edge count matches exactly (190), we must use EVERY edge in K20 exactly once.
    # This is a tree decomposition of K20.
    
    # Strategy:
    # 1. Randomly shuffle the order of trees to break symmetry? 
    #    Or keep fixed order but randomize vertex permutations.
    # 2. Backtrack on tree index.
    # 3. For each tree, try random permutations of vertices 0..19 of length k.
    #    Check if the induced edges are available.
    
    # To make it faster, we can use a heuristic:
    # Embed trees in order of decreasing number of edges (largest first).
    # This reduces the search space early because large trees have fewer valid placements 
    # in a sparse remaining graph (though initially the graph is dense).
    # Actually, in a dense graph, large trees have MANY valid placements.
    # In a sparse graph (later stages), small trees have many placements, large trees have few.
    # So, embedding large trees first is generally better for CSPs on graphs.
    
    # Let's sort trees by size descending: T20, T19, ..., T1.
    # Indices: 19, 18, ..., 0.
    
    order = list(range(19, -1, -1))
    
    best_score = 0
    best_result = [None] * 20
    
    # We will run multiple random restarts.
    # In each restart, we pick a random permutation of the order? 
    # Or just random vertex mappings?
    # Random vertex mappings are the primary source of non-determinism.
    
    def solve(order, rng):
        # used_edges: set of frozensets or tuples
        used_edges = set()
        mappings = [None] * 20
        score = 0
        
        # Recursive backtracking
        # We need to be careful with recursion depth (20 is fine) and time.
        
        def backtrack(idx):
            nonlocal used_edges, mappings, score
            
            if time.time() > deadline:
                return False
            
            if idx == len(order):
                return True
            
            k_idx = order[idx]
            k = k_idx + 1
            edges = trees[k_idx]
            adj = tree_adj[k_idx]
            
            if k == 1:
                # T1: single vertex, no edges. Can map to any unused vertex?
                # Wait, the problem says "list of k distinct int host vertices".
                # It doesn't explicitly say vertices must be distinct across trees, 
                # only that edges must be edge-disjoint.
                # "Embedded trees must be edge-disjoint in K20."
                # Vertex sharing is allowed.
                # So T1 can map to any vertex, say 0.
                mappings[k_idx] = [0]
                # No edges to add
                if backtrack(idx + 1):
                    return True
                mappings[k_idx] = None
                return False
            
            # For k > 1, we need to map k distinct vertices.
            # Number of permutations of k vertices from 20 is P(20, k).
            # For k=20, it's 20! which is huge.
            # We cannot iterate all permutations.
            
            # Heuristic: 
            # If k == 20, the mapping must be a permutation of 0..19.
            # The edges of T20 must be a subset of available edges.
            # Since we start with all edges available, T20 can be ANY spanning tree.
            # But T20 is a FIXED tree structure.
            # We need to check if the fixed tree T20 can be embedded in K20.
            # Since K20 is complete, ANY tree on 20 vertices can be embedded.
            # So for T20, we just need to pick a random permutation of 0..19.
            # But we must ensure that the edges used by T20 don't conflict with future trees.
            # Since we are doing backtracking, we just pick one, mark edges, and proceed.
            
            # Optimization: 
            # Instead of trying all permutations, we can try a few random ones.
            # If we fail, we backtrack.
            # But backtracking over 20 levels with random restarts inside might be slow.
            
            # Alternative: 
            # Use a greedy-with-random-restarts approach for the whole pack, 
            # but structured as a CSP to allow backtracking if a greedy choice leads to a dead end.
            
            # Let's try a limited number of random permutations for this tree.
            # If k is large, the number of valid embeddings in the remaining graph might be small.
            # If k is small, it's large.
            
            # Let's try up to 50 random permutations for each tree node.
            # If none work, return False (triggering backtrack to parent).
            
            attempts = 0
            max_attempts = 100 if k > 10 else 50
            
            while attempts < max_attempts:
                attempts += 1
                if time.time() > deadline:
                    return False
                
                # Generate random mapping for this tree
                # Vertices must be distinct.
                verts = rng.sample(range(20), k)
                # Map tree vertex i to verts[i]
                # Check edges
                valid = True
                used_here = []
                for u, v in edges:
                    hu = verts[u]
                    hv = verts[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                    used_here.append(e_key)
                
                if valid:
                    # Apply changes
                    used_edges.update(used_here)
                    mappings[k_idx] = verts
                    score += len(edges)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo changes
                    for e in used_here:
                        used_edges.remove(e)
                    mappings[k_idx] = None
                    score -= len(edges)
                    
                    if time.time() > deadline:
                        return False
            
            return False
        
        if backtrack(0):
            return mappings, score
        return None, 0

    # Run multiple restarts
    # Each restart uses a different random seed for the permutations
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        rng = Random(611 + restarts)
        # Shuffle the order slightly? 
        # Maybe keep T20 first always, as it's the most constrained in terms of vertex count.
        # Or shuffle the smaller trees?
        # Let's stick to descending size for now, but randomize the vertex mappings.
        
        res, sc = solve(order, rng)
        if res is not None:
            if sc > best_score:
                best_score = sc
                best_result = res
                # If we found a perfect packing (190 edges), we can stop.
                if best_score == 190:
                    break
        else:
            # If solve returns None, it means it didn't finish or failed.
            # We should capture the best partial result?
            # The current solve() only returns if it completes fully.
            # We need a way to return the best partial result found during the search.
            
            # Let's modify solve to track best partial score and return it if deadline is hit.
            # This is complex. 
            # Alternative: 
            # The previous greedy approach (score 6.8) just greedily packed as many as it could.
            # My backtracking approach might take a long time to find a full solution if one doesn't exist easily.
            # If a full solution exists, backtracking will find it.
            # If not, it will exhaust the search or hit the deadline.
            
            # To be safe and competitive with the greedy baseline, 
            # I should also try to extract a partial solution.
            # But the current `solve` structure doesn't easily expose the best partial state.
            
            # Let's refactor: 
            # Instead of pure backtracking, let's do a "Greedy with Backtracking" hybrid?
            # Or just rely on the fact that if a perfect packing exists, 
            # random restarts of backtracking should find it quickly?
            
            # Given the tight constraint (190 edges), a perfect packing is likely the target.
            # If I can't find it, I might score 0, which is worse than 6.8.
            
            # Let's add a fallback: 
            # If backtracking fails to find a full solution within a short time, 
            # fall back to a simple greedy heuristic to ensure a non-zero score.
            
            pass

    # Fallback: Simple Greedy
    # If best_score is low, try a greedy approach.
    if best_score < 50:
        greedy_score = 0
        greedy_result = [None] * 20
        used_edges = set()
        
        # Embed trees from largest to smallest
        for k_idx in range(19, -1, -1):
            k = k_idx + 1
            edges = trees[k_idx]
            if k == 1:
                greedy_result[k_idx] = [0]
                continue
            
            found = False
            for _ in range(1000): # 1000 attempts
                verts = random.sample(range(20), k)
                valid = True
                for u, v in edges:
                    hu = verts[u]
                    hv = verts[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                if valid:
                    for u, v in edges:
                        hu = verts[u]
                        hv = verts[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    greedy_result[k_idx] = verts
                    greedy_score += len(edges)
                    found = True
                    break
        
        if greedy_score > best_score:
            best_score = greedy_score
            best_result = greedy_result

    # Ensure types are correct
    final_result = []
    for r in best_result:
        if r is None:
            final_result.append(None)
        else:
            # Ensure it's a list of ints
            final_result.append([int(x) for x in r])
            
    return final_result

# EVOLVE-BLOCK-END
