import time
import random
from itertools import permutations

# Re-implement instance_trees to ensure consistency and availability
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
            # Find the smallest leaf
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        # The last two vertices with degree 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

# Precompute the trees
TREES = instance_trees(20, 611)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 110s to be safe
    
    # We want to pack T_1 ... T_20
    # Let's try to pack them in order of size, maybe largest first to constrain early?
    # Or smallest first?
    # Let's try largest first: 20, 19, ..., 1
    # Actually, T_1 is trivial. T_2 is 1 edge.
    # Let's just do 1..20.
    
    best_score = 0
    best_result = [None] * 20
    
    # Helper to check if a set of edges is disjoint from used_edges
    def is_disjoint(tree_edges, used_edges):
        for e in tree_edges:
            if e in used_edges or (e[1], e[0]) in used_edges:
                return False
        return True

    def backtrack(k, used_edges, embeddings):
        nonlocal best_score, best_result
        
        if time.time() > deadline:
            return False
            
        if k == 20:
            # All trees embedded
            score = sum(len(TREES[i]) for i in range(20))
            if score > best_score:
                best_score = score
                # Copy the result
                best_result = [list(e) if e is not None else None for e in embeddings]
            return True
            
        tree = TREES[k]
        num_vertices = k + 1 # Tree T_k has k+1 vertices? 
        # Wait, the task says "Entry k-1 is None or a list of k distinct int host vertices... position i is the image of vertex i of the fixed tree T_k."
        # Let's check the generation code.
        # for k in range(1, n+1):
        #   if k<=2: ...
        #   code = [rng.randrange(k) for _ in range(k-2)]
        #   degree = [1]*k
        #   ...
        #   edges ...
        # The trees are generated for indices 0..k-1?
        # "position i is the image of vertex i of the fixed tree T_k"
        # In the generation loop `for k in range(1, n+1)`, the variable `k` is used for size.
        # If k=1, tree has 1 vertex? No, `if k==1: trees.append([])`. 0 edges. 1 vertex.
        # If k=2, `trees.append([(0,1)])`. 1 edge. 2 vertices.
        # If k=3, `code` has 1 element. `degree` size 3.
        # So T_k has k vertices?
        # Let's re-read carefully.
        # "Entry k-1 is None or a list of k distinct int host vertices"
        # So T_1 has 1 vertex. T_2 has 2 vertices. T_k has k vertices.
        # The generation code:
        # for k in range(1, n+1):
        #   if k <= 2: ...
        #   code = [rng.randrange(k) for _ in range(k-2)]
        #   degree = [1] * k
        #   ...
        #   edges ...
        #   trees.append(edges)
        # Yes, T_k has k vertices (labeled 0 to k-1 in the tree's own coordinate system).
        
        # So for index `k` in 0..19, the tree is `TREES[k]` which has `k+1` vertices.
        # Wait, the loop in instance_trees is `for k in range(1, n+1)`.
        # So `TREES[0]` corresponds to `k=1` (1 vertex).
        # `TREES[1]` corresponds to `k=2` (2 vertices).
        # `TREES[k]` corresponds to tree size `k+1`.
        
        # My backtrack index `k` goes from 0 to 19.
        # So `TREES[k]` has `k+1` vertices.
        
        # We need to assign `k+1` distinct vertices from 0..19 to the tree's vertices 0..k.
        # This is a subset selection + permutation.
        
        # Heuristic: Try to find a valid embedding for TREES[k]
        # We can iterate over combinations of vertices? 20C(k+1) is too big for large k.
        # Better: Randomized backtracking on the vertices of the tree.
        
        # Let's restructure backtrack to assign vertices one by one for the current tree.
        # This is complex to implement generically in a single function without passing state around.
        
        # Alternative: Use a dedicated function for embedding one tree given used edges.
        pass

    # Let's implement a specific embedding function for a tree T into K20 avoiding used_edges.
    # We'll use a randomized DFS on the tree's vertices.
    
    def embed_tree(tree_idx, used_edges, rng):
        """
        Attempts to embed TREES[tree_idx] into K20 avoiding used_edges.
        Returns a list of length (tree_idx+1) where entry i is the host vertex for tree vertex i, or None if failed.
        """
        k = tree_idx + 1 # Number of vertices in the tree
        tree_edges = TREES[tree_idx]
        
        # We need to assign host vertices to tree vertices 0..k-1.
        # Let's pick a random starting vertex for tree vertex 0.
        # Then traverse the tree, assigning random available vertices.
        
        # To make it more likely to succeed, we should probably check constraints as we go.
        # But checking edge disjointness requires knowing both endpoints.
        
        # Let's use a backtracking search for the embedding of this single tree.
        # State: mapping from tree_vertex -> host_vertex.
        # We can order the tree vertices by degree or just use a DFS order.
        
        # Build adjacency list for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Find a root (vertex 0 is fine, or any)
        # Let's pick a random permutation of tree vertices to assign?
        # No, tree structure matters.
        
        # Let's try to assign vertices in an order that allows early failure.
        # High degree vertices first?
        # Or just DFS from 0.
        
        # We will use a recursive helper.
        def dfs(tree_v, host_v, mapping, used_hosts):
            if time.time() > deadline:
                return None
            if len(mapping) == k:
                # Check all edges
                for u, v in tree_edges:
                    hu, hv = mapping[u], mapping[v]
                    if (hu, hv) in used_edges or (hv, hu) in used_edges:
                        return None
                return mapping
            
            # Try to assign neighbors
            # Actually, we just need to ensure that when we assign a neighbor, the edge is valid.
            # But we don't know the neighbor's host yet.
            
            # Better approach: Assign all vertices first, then check edges?
            # That's too many permutations.
            
            # Hybrid: Assign vertices in DFS order.
            # When assigning a child, we know the parent's host.
            # We can check if the edge (parent_host, child_host) is in used_edges.
            # This is valid!
            
            # We need to ensure all host vertices are distinct.
            
            # Let's restart the DFS logic properly.
            pass
        
        # The above DFS needs to be careful.
        # Let's implement a proper backtracker for the single tree embedding.
        
        # Order of assignment: 0, then neighbors of 0, etc.
        # We need a list of tree vertices to visit.
        # Let's just use a simple randomized backtracking on the tree vertices.
        
        # To speed up, we can pre-shuffle the order of tree vertices?
        # No, the tree structure is fixed.
        
        # Let's try a different strategy:
        # Since K20 is dense, random assignment of distinct vertices is very likely to be valid
        # unless the used_edges are very dense.
        # We can just pick k random distinct vertices from 0..19, and a random permutation,
        # and check if it's valid. If not, retry.
        # Given 190 total edges and we use at most 190, the probability of collision is low initially.
        
        max_attempts = 1000
        for _ in range(max_attempts):
            # Pick k distinct host vertices
            hosts = rng.sample(range(20), k)
            # Randomly map tree vertices 0..k-1 to these hosts
            perm = hosts[:]
            rng.shuffle(perm)
            
            # Check validity
            valid = True
            for u, v in tree_edges:
                hu, hv = perm[u], perm[v]
                if (hu, hv) in used_edges or (hv, hu) in used_edges:
                    valid = False
                    break
            
            if valid:
                return perm
        
        return None

    # Now the main loop
    # We will embed trees one by one.
    # Order: Let's try 20 down to 1 (largest first).
    # Or 1 to 20.
    # Largest first might fail sooner if it can't find space, but it uses many edges.
    # Smallest first might leave fragmented space?
    # Let's try 20 down to 1.
    
    order = list(range(19, -1, -1)) # 19 (T20) down to 0 (T1)
    
    # To increase success rate, we can do multiple restarts.
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(r) # Deterministic restarts for reproducibility, or just random
        # Actually, using a new Random() each time is fine.
        
        used_edges = set()
        embeddings = [None] * 20
        success = True
        
        for idx in order:
            # idx is 0..19, corresponding to T_{idx+1}
            # But wait, my `embed_tree` expects `tree_idx` which is 0..19.
            # `TREES[idx]` has `idx+1` vertices.
            
            res = embed_tree(idx, used_edges, rng)
            if res is None:
                success = False
                break
            
            embeddings[idx] = res
            
            # Add edges to used_edges
            for u, v in TREES[idx]:
                hu, hv = res[u], res[v]
                used_edges.add((hu, hv))
                # Note: we store directed or undirected?
                # `is_disjoint` checks both. `embed_tree` checks both.
                # So storing one direction is enough if we check both in collision.
                # But it's safer to store both? No, set of tuples is fine.
                # Just ensure consistency.
                
        if success:
            # We found a full packing!
            # Score is 190.
            # Convert to required format: list of 20 entries.
            # Entry k-1 is list of k vertices?
            # "Entry k-1 is None or a list of k distinct int host vertices"
            # My `embeddings` list has index `idx` corresponding to `TREES[idx]` which has `idx+1` vertices.
            # So `embeddings[0]` has 1 vertex (for T1).
            # `embeddings[19]` has 20 vertices (for T20).
            # This matches the requirement.
            
            # Check types: integers must be Python int.
            # `rng.sample` returns ints. `rng.shuffle` keeps them as ints.
            # So they are Python ints.
            
            return embeddings

    # If no full packing found, return the best partial one?
    # The task says "returning 20 entries... raw value is total number of embedded tree edges".
    # If we fail to embed a tree, it's None.
    # We should return the best partial solution found.
    
    # Let's modify the loop to track the best partial solution.
    # Actually, the loop above breaks on failure.
    # We can restart and try to improve.
    
    # For now, if we didn't return, we return the best we found in the last restart?
    # Or we can keep a global best.
    
    # Let's refine:
    # We'll run the restarts and keep the best score.
    
    best_score = 0
    best_result = [None] * 20
    
    for r in range(500): # More restarts
        if time.time() > deadline:
            break
            
        rng = random.Random(r)
        used_edges = set()
        embeddings = [None] * 20
        current_score = 0
        
        for idx in order:
            res = embed_tree(idx, used_edges, rng)
            if res is None:
                break
            embeddings[idx] = res
            for u, v in TREES[idx]:
                hu, hv = res[u], res[v]
                used_edges.add((hu, hv))
            current_score += len(TREES[idx])
            
        if current_score > best_score:
            best_score = current_score
            best_result = [list(e) if e is not None else None for e in embeddings]
            
            if best_score == 190:
                return best_result

    return best_result

# To ensure we don't hang, we add a final check
# If best_result is all None, return [None]*20
# But T1 is always embeddable (0 edges).
# So best_score should be at least 0.

# Let's double check the vertex count logic.
# T1 (idx 0): 1 vertex.
# T2 (idx 1): 2 vertices.
# ...
# T20 (idx 19): 20 vertices.
# The output format is a list of 20 entries.
# Entry 0 corresponds to T1.
# Entry 19 corresponds to T20.
# This matches.

# One detail: `embed_tree` uses `rng.sample(range(20), k)`.
# For k=20, this is a permutation of 0..19.
# For k=1, this is a single element.
# This is correct.

# Potential issue: `embed_tree` might be slow if it fails many times.
# `max_attempts = 1000` is a hard cap per tree per restart.
# If it fails, we break the restart.
# This is efficient.

# Let's wrap it in the final function.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute trees
    TREES = instance_trees(20, 611)
    
    # Helper to embed a single tree
    def embed_tree(tree_idx, used_edges, rng):
        k = tree_idx + 1
        tree_edges = TREES[tree_idx]
        if not tree_edges:
            # T1, just pick any vertex
            return [rng.randint(0, 19)]
            
        max_attempts = 2000
        for _ in range(max_attempts):
            hosts = rng.sample(range(20), k)
            perm = hosts[:]
            rng.shuffle(perm)
            
            valid = True
            for u, v in tree_edges:
                hu, hv = perm[u], perm[v]
                if (hu, hv) in used_edges or (hv, hu) in used_edges:
                    valid = False
                    break
            
            if valid:
                return perm
        return None

    order = list(range(19, -1, -1))
    best_score = 0
    best_result = [None] * 20
    
    # Try a few fixed seeds first for determinism if possible, then random
    for r in range(1000):
        if time.time() > deadline:
            break
            
        rng = random.Random(r)
        used_edges = set()
        embeddings = [None] * 20
        current_score = 0
        
        for idx in order:
            res = embed_tree(idx, used_edges, rng)
            if res is None:
                break
            embeddings[idx] = res
            for u, v in TREES[idx]:
                hu, hv = res[u], res[v]
                used_edges.add((hu, hv))
            current_score += len(TREES[idx])
            
        if current_score > best_score:
            best_score = current_score
            best_result = [list(e) if e is not None else None for e in embeddings]
            
            if best_score == 190:
                return best_result

    return best_result
