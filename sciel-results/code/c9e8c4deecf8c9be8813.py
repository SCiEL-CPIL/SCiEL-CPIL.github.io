import time
import sys
from random import Random

def _generate_trees(n=20, seed=611):
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
    deadline = time.time() + 140  # safety margin before 150s kill
    trees = _generate_trees(20, 611)
    n = 20
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # We need to assign each tree's edges to distinct K20 edges
    # Process trees in order of decreasing size (more constrained first)
    # Tree k has k-1 edges, so process k=20,19,...,1
    
    # For each tree, we need to find an injective mapping from its vertices to 0..19
    # such that all edges map to available K20 edges
    
    # Use backtracking with the remaining available edges
    # Key: embed trees one by one, removing used edges
    
    # Order: larger trees first
    order = list(range(19, -1, -1))  # tree indices 19 (k=20) down to 0 (k=1)
    
    # For each tree, we'll try to embed it
    # A tree with k vertices has k-1 edges
    # We need to assign k distinct host vertices from 0..19
    
    # Strategy: use recursive backtracking per tree, choosing vertex assignments
    # that respect edge constraints with remaining edges
    
    result = [None] * 20
    available = set(all_edges)
    
    def can_embed(tree_idx, available_edges):
        """Try to find embedding for tree tree_idx in available_edges.
        Returns the embedding (list of k host vertices) or None."""
        k = tree_idx + 1
        if k == 1:
            # Single vertex, any host vertex works
            return [0]
        if k == 2:
            # Edge (0,1), need an available edge
            for (a, b) in available_edges:
                return [a, b]
            return None
        
        tree_edges = trees[tree_idx]
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (u, v) in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find a leaf to start from (degree 1)
        # Actually, let's use a root-based approach
        # Pick vertex 0 as root, do BFS/DFS to get ordering
        
        # Find a good starting vertex - a leaf
        leaf = None
        for v in range(k):
            if len(adj[v]) == 1:
                leaf = v
                break
        if leaf is None:
            leaf = 0
        
        # Build parent-child structure rooted at leaf
        parent = [-1] * k
        children = [[] for _ in range(k)]
        visited = [False] * k
        visited[leaf] = True
        stack = [leaf]
        order_list = [leaf]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    children[u].append(v)
                    stack.append(v)
                    order_list.append(v)
        
        # Now embed: assign host vertices
        # Start with leaf mapped to some host vertex h
        # Then for each child, must map to a host vertex adjacent to h via available edge
        
        # This is still complex. Let's use a different approach:
        # Since the tree is small (max 20 vertices), use constraint propagation
        
        # Try all possible assignments for the first vertex, then propagate
        # But that's 20! possibilities...
        
        # Better: use the fact that we can pick any host vertex for the root,
        # then for each child, the host must be a neighbor in available graph
        
        # Let's build the available graph's adjacency
        avail_adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            avail_adj[a].add(b)
            avail_adj[b].add(a)
        
        # Backtrack: assign host vertices in order_list order
        host = [-1] * k
        host[leaf] = -1  # to be assigned
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == len(order_list):
                # All assigned, verify all tree edges are in available
                for (u, v) in tree_edges:
                    hu, hv = host[u], host[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e not in available_edges:
                        return None
                return list(host)
            
            v = order_list[idx]
            # Determine which host vertices are possible for v
            if parent[v] == -1:
                # Root, any host vertex
                candidates = list(range(n))
            else:
                # Must be adjacent to parent's host in available graph
                pv = parent[v]
                if host[pv] == -1:
                    return None
                candidates = list(avail_adj[host[pv]])
            
            # Try each candidate
            for h in candidates:
                if h in [host[x] for x in order_list[:idx]]:
                    continue  # already used
                host[v] = h
                res = backtrack(idx + 1)
                if res is not None:
                    return res
            host[v] = -1
            return None
        
        return backtrack(0)
    
    # Process trees in order
    success = True
    for t_idx in order:
        if time.time() > deadline:
            # Return what we have
            break
        k = t_idx + 1
        if k <= 2:
            # Easy cases
            if k == 1:
                # Find any available host vertex
                for v in range(n):
                    result[t_idx] = [v]
                    break
            else:
                # k=2, need an available edge
                for (a, b) in available:
                    result[t_idx] = [a, b]
                    available.discard((a, b))
                    break
                if result[t_idx] is None:
                    success = False
                    break
            continue
        
        emb = can_embed(t_idx, available)
        if emb is None:
            success = False
            break
        result[t_idx] = emb
        # Remove used edges
        for (u, v) in trees[t_idx]:
            hu, hv = emb[u], emb[v]
            e = (min(hu, hv), max(hu, hv))
            if e in available:
                available.discard(e)
            else:
                # Should not happen if can_embed verified
                success = False
                break
        if not success:
            break
    
    # Ensure all entries are proper lists of ints or None
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    # Count total embedded edges
    total_edges = 0
    for i in range(20):
        if result[i] is not None:
            total_edges += len(trees[i])
    
    return result
