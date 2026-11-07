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
    deadline = start_time + 140  # 140s out of 150s
    
    trees = instance_trees(20, 611)
    
    # We want to embed trees T1..T20 (index 0..19)
    # T1 has 0 edges, T2 has 1, ..., T20 has 19
    
    # Strategy: embed trees in order of decreasing size (most constrained first)
    # But T1 is trivial (no edges), T2 is simple, etc.
    
    # Actually, let's try a greedy approach with backtracking for each tree
    
    # forbidden_edges: set of frozenset({u,v}) for edges already used
    forbidden = set()
    embeddings = [None] * 20  # embeddings[k-1] is the list of host vertices for Tk
    
    # For each tree, we need to find an embedding
    # Order: try larger trees first? Or smaller? 
    # Larger trees use more edges, so they're more constrained. Let's do larger first.
    
    # Actually, T1 has 0 edges - always embeddable (just pick any vertex)
    # T2 has 1 edge - need 2 distinct vertices with an edge between them (always possible in K20)
    # ...
    # T20 has 19 edges - needs all 20 vertices
    
    # Let's process in order 20, 19, 18, ..., 1 (largest first)
    # But we need to return in order 1..20
    
    order = list(range(19, -1, -1))  # indices 19, 18, ..., 0 (T20, T19, ..., T1)
    
    # For backtracking embedding of a tree
    def embed_tree(tree_idx):
        """Try to embed tree tree_idx into K20 avoiding forbidden edges.
        Returns a list of 20 host vertices (position i = image of vertex i) or None."""
        k = tree_idx + 1
        edges = trees[tree_idx]
        
        if k == 1:
            # No edges, just pick any available vertex
            # Actually, we just need to assign vertex 0 to some host
            # But since there are no edges, any assignment works
            # We need to return a list of 1 element? No, the spec says "list of k distinct int host vertices"
            # For k=1, it's a list of 1 element
            # But wait, the spec says "position i is the image of vertex i of the fixed tree T_k"
            # So for T1, we return [v] for some v in 0..19
            # But actually, looking at the seed: it returns None for T1..T19
            # So we can return None for T1 if we want, but let's embed it for completeness
            # Actually, T1 contributes 0 edges, so it doesn't matter for the raw value
            # Let's just return None for T1 to keep it simple, or embed it
            # Let's embed it: pick any vertex
            # But we need to track which vertices are used? No, vertices can be shared!
            # Only edges must be disjoint. Vertices can be reused across trees.
            # So for T1, any single vertex works.
            return [0]  # or any vertex
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find a root and do BFS/DFS to get an ordering
        # We'll use backtracking: assign tree vertices one by one
        # Order: BFS from a root
        
        # Find a good root: maybe the vertex with highest degree
        # Or just use vertex 0
        root = 0
        visited = [False] * k
        order_list = [root]
        visited[root] = True
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order_list.append(v)
                    queue.append(v)
        
        # Now order_list is a BFS ordering
        # We'll assign in this order
        # assignment[i] = host vertex for tree vertex order_list[i]
        
        # Backtracking
        def backtrack(pos, assignment):
            if time.time() > deadline:
                return None
            
            if pos == k:
                # Check all edges
                for u, v in edges:
                    a, b = assignment[u], assignment[v]
                    edge = frozenset((a, b))
                    if edge in forbidden:
                        return None
                return assignment.copy()
            
            tree_v = order_list[pos]
            # Try each host vertex
            for host in range(20):
                if host in assignment.values():
                    continue  # vertex already used
                
                # Check edges from tree_v to already-assigned neighbors
                valid = True
                for nb in adj[tree_v]:
                    if nb in assignment:
                        edge = frozenset((host, assignment[nb]))
                        if edge in forbidden:
                            valid = False
                            break
                
                if valid:
                    assignment[tree_v] = host
                    result = backtrack(pos + 1, assignment)
                    if result is not None:
                        return result
                    del assignment[tree_v]
            
            return None
        
        result = backtrack(0, {})
        if result is not None:
            # Convert to list: position i = image of vertex i
            return [result[i] for i in range(k)]
        return None
    
    # Process trees in order
    for tree_idx in order:
        if time.time() > deadline:
            break
        
        k = tree_idx + 1
        if k == 1:
            embeddings[tree_idx] = [0]
            continue
        
        result = embed_tree(tree_idx)
        if result is not None:
            embeddings[tree_idx] = result
            # Add edges to forbidden
            for u, v in trees[tree_idx]:
                a, b = result[u], result[v]
                forbidden.add(frozenset((a, b)))
        else:
            # Couldn't embed this tree
            # Try with randomization? Or skip it
            embeddings[tree_idx] = None
    
    # Count total edges
    total = 0
    for i in range(20):
        if embeddings[i] is not None:
            total += i  # Ti has i edges (T1 has 0, T2 has 1, ...)
            # Wait, T_k has k-1 edges, so T_i (index i-1) has i-1 edges
            # Actually, trees[i] is T_{i+1}, which has i edges
            # Let me recount: trees[0] = T1 (0 edges), trees[1] = T2 (1 edge), ..., trees[19] = T20 (19 edges)
            # So trees[i] has i edges
            total = 0  # reset
            for j in range(20):
                if embeddings[j] is not None:
                    total += j  # trees[j] has j edges
    
    return embeddings
