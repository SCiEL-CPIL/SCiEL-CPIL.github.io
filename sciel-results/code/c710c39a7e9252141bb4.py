import time
import random
from random import Random
from collections import deque

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
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency and degree for each tree
    tree_adj = []
    tree_deg = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Sort trees by size descending for packing order
    order = list(range(n))
    order.sort(key=lambda x: len(trees[x]), reverse=True)
    
    # Try multiple random restarts with different random seeds
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 678)
        
        available = set(all_edges)
        current_solution = [None] * n
        success = True
        packed_edges = 0
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                # T1: Single vertex, no edges
                v = rng.randint(0, n - 1)
                current_solution[t_idx] = [v]
                continue
            
            # Build adjacency for this tree
            adj = tree_adj[t_idx]
            deg = tree_deg[t_idx]
            
            # Pick a random root - prefer high-degree nodes
            if rng.random() < 0.8:
                # Pick a high-degree node as root
                max_deg = max(deg)
                high_deg_nodes = [i for i in range(k) if deg[i] == max_deg]
                root = rng.choice(high_deg_nodes)
            else:
                root = rng.randint(0, k - 1)
            
            # Try to embed using backtracking with MRV heuristic
            mapping = [-1] * k
            used_vertices = [False] * n
            
            # Precompute for each tree vertex: which host vertices could map to it
            # based on degree constraints (a tree vertex of degree d needs at least d available edges)
            
            def get_candidates_for_node(node, parent):
                """Get candidate host vertices for a tree node, using MRV-style filtering."""
                if parent != -1:
                    parent_gv = mapping[parent]
                    # Must be neighbor of parent_gv with available edge
                    candidates = []
                    for v in range(n):
                        if v == parent_gv or used_vertices[v]:
                            continue
                        e = (min(parent_gv, v), max(parent_gv, v))
                        if e in available:
                            candidates.append(v)
                else:
                    candidates = [v for v in range(n) if not used_vertices[v]]
                
                return candidates
            
            def count_available_edges(v):
                """Count how many available edges incident to vertex v."""
                count = 0
                for u in range(n):
                    if u == v:
                        continue
                    e = (min(v, u), max(v, u))
                    if e in available:
                        count += 1
                return count
            
            def embed_dfs(node, parent):
                nonlocal available, used_vertices
                if time.time() > deadline:
                    return False
                
                candidates = get_candidates_for_node(node, parent)
                
                if not candidates:
                    return False
                
                # Score candidates: prefer those with fewer available edges (more constrained)
                # but also consider the degree requirement of the tree node
                node_deg = deg[node]
                scored_candidates = []
                for gv in candidates:
                    # How many available edges does gv have?
                    avail = count_available_edges(gv)
                    # We need node_deg edges from gv (one to parent if parent != -1, rest to children)
                    needed = node_deg - (1 if parent != -1 else 0)
                    # Score: prefer vertices that can satisfy the degree requirement
                    # and have fewer excess edges (to preserve flexibility)
                    if avail < needed:
                        # Can't satisfy degree requirement - skip
                        continue
                    # Prefer vertices with avail closer to needed (less waste)
                    score = avail - needed
                    scored_candidates.append((score, gv))
                
                if not scored_candidates:
                    return False
                
                # Sort by score (prefer lower excess), then shuffle for randomness
                scored_candidates.sort(key=lambda x: x[0])
                # Add some randomness among close scores
                for i in range(len(scored_candidates)):
                    if rng.random() < 0.3:
                        j = rng.randint(0, min(i, 3))
                        scored_candidates[i], scored_candidates[j] = scored_candidates[j], scored_candidates[i]
                
                for _, gv in scored_candidates:
                    mapping[node] = gv
                    used_vertices[gv] = True
                    ok = True
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if not embed_dfs(neighbor, node):
                            ok = False
                            break
                    
                    if ok:
                        return True
                    
                    mapping[node] = -1
                    used_vertices[gv] = False
                
                return False
            
            if not embed_dfs(root, -1):
                # Failed to embed this tree
                success = False
                break
            
            # Update available edges
            for u, v in trees[t_idx]:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                available.remove(e)
                packed_edges += 1
                
            current_solution[t_idx] = mapping[:]
        
        if success:
            # Found a complete packing
            return current_solution
    
    # If no complete solution found, return a fallback
    result = [None] * n
    result[0] = [0]
    result[1] = [0, 1]
    
    return result
