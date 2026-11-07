from random import Random
import time
import sys

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees


def _embed_tree(tree_edges, k, available_edges, rng):
    """
    Try to embed tree with k vertices and given edges into available_edges.
    Returns list of k host vertices (image of tree vertex i), or None.
    available_edges is a set of frozenset({u,v}) for u!=v.
    """
    if k == 1:
        # Single vertex, just pick any available vertex
        # But we need to track which vertices are used
        # Actually for k=1, any single vertex works
        return [0]  # placeholder, will be handled specially
    
    if k == 2:
        # Need an edge
        for e in available_edges:
            a, b = tuple(e)
            return [a, b]
        return None
    
    # Build adjacency for the tree
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Root the tree at 0
    parent = [-1] * k
    order = [0]
    parent[0] = -2  # mark as root
    for node in order:
        for nb in adj[node]:
            if parent[nb] == -1:
                parent[nb] = node
                order.append(nb)
    
    # Try randomized greedy with some backtracking
    # We'll do multiple random attempts
    for attempt in range(200):
        # Shuffle the order of children at each node for randomness
        child_order = {}
        for node in range(k):
            children = [nb for nb in adj[node] if parent[nb] == node]
            rng.shuffle(children)
            child_order[node] = children
        
        assignment = [None] * k
        # Assign root
        # Pick a random vertex from those with enough available edges
        # For simplicity, try a random vertex
        host_vertices = list(range(20))
        rng.shuffle(host_vertices)
        
        success = True
        # BFS assignment
        queue = [0]
        assignment[0] = rng.choice(host_vertices)
        used_hosts = {assignment[0]}
        
        for node in order[1:]:
            par = parent[node]
            par_host = assignment[par]
            # Find available neighbors of par_host
            candidates = []
            for e in available_edges:
                if par_host in e:
                    other = next(iter(e - {par_host}))
                    if other not in used_hosts:
                        candidates.append(other)
            if not candidates:
                success = False
                break
            # Prefer candidates with fewer options (for children)
            # Simple: pick random
            assignment[node] = rng.choice(candidates)
            used_hosts.add(assignment[node])
        
        if success:
            # Verify all edges
            valid = True
            for u, v in tree_edges:
                eu = frozenset((assignment[u], assignment[v]))
                if eu not in available_edges:
                    valid = False
                    break
            if valid:
                return assignment
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    trees = instance_trees(20, 611)
    
    # We want to embed trees in order of decreasing size
    # Tree k has k vertices and k-1 edges (for k>=2), tree 1 has 0 edges
    # Total edges = sum(k-1 for k=1..20) = 0+1+2+...+19 = 190
    
    # Strategy: embed large trees first
    # Order: 20, 19, 18, ..., 2, 1
    
    best_result = None
    best_score = -1
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        rng = Random(restart * 1000 + 42)
        
        # Available edges: all edges of K20
        available = set()
        for u in range(20):
            for v in range(u + 1, 20):
                available.add(frozenset((u, v)))
        
        result = [None] * 20
        success = True
        
        # Embed trees from largest to smallest
        for k in range(20, 0, -1):
            if time.time() > deadline:
                success = False
                break
            
            tree_edges = trees[k - 1]
            num_edges = len(tree_edges)
            
            if k == 1:
                # Single vertex, no edges. Any unused vertex works.
                used = set()
                for i in range(20):
                    if result[i] is not None:
                        used.update(result[i])
                free = [v for v in range(20) if v not in used]
                if not free:
                    success = False
                    break
                result[k - 1] = [rng.choice(free)]
                continue
            
            if k == 2:
                # Need one available edge
                if not available:
                    success = False
                    break
                e = rng.choice(list(available))
                a, b = tuple(e)
                result[k - 1] = [a, b]
                available.discard(e)
                continue
            
            # General case
            emb = _embed_tree(tree_edges, k, available, rng)
            if emb is None:
                success = False
                break
            
            result[k - 1] = emb
            # Remove used edges from available
            for u, v in tree_edges:
                e = frozenset((emb[u], emb[v]))
                if e not in available:
                    # Should not happen if embedding is valid
                    success = False
                    break
                available.discard(e)
            if not success:
                break
        
        if not success:
            continue
        
        # Calculate score
        score = sum(len(trees[i]) for i in range(20))
        if score > best_score:
            best_score = score
            best_result = result
    
    if best_result is not None:
        # Ensure all entries are proper lists of ints
        final = []
        for i in range(20):
            if best_result[i] is None:
                final.append(None)
            else:
                final.append([int(v) for v in best_result[i]])
        return final
    else:
        # Fallback: only T20 via identity
        return [None] * 19 + [list(range(20))]
