import time
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

def _find_embedding(tree_edges, n, used_edges, deadline):
    """
    Find an embedding of tree_edges into K_n avoiding used_edges.
    Returns a list of length len(tree_edges)+1 mapping tree vertex -> host vertex, or None.
    """
    if not tree_edges:
        return []
    
    # Build adjacency for the tree
    k = max(max(u, v) for u, v in tree_edges) + 1
    tree_adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        tree_adj[u].append(v)
        tree_adj[v].append(u)
    
    # Find a leaf to start from
    leaves = [i for i in range(k) if len(tree_adj[i]) == 1]
    if not leaves:
        # Single vertex tree?
        return [0]
    
    # Try different starting leaves
    start_leaf = leaves[0]
    
    # Backtracking search
    # mapping: tree_vertex -> host_vertex
    mapping = [-1] * k
    used_hosts = set()
    
    # We'll do DFS with backtracking
    # Order: start from a leaf, grow the tree
    # We need to track which tree vertices are mapped
    
    # For efficiency, we'll use a specific ordering:
    # Start with start_leaf mapped to some host, then BFS/DFS the tree
    
    # Let's use a recursive backtracking approach
    # State: which tree vertices are mapped, and their host assignments
    
    # To make it faster, we'll fix the first vertex (start_leaf) to vertex 0
    # and try all permutations for the rest? No, that's too slow.
    
    # Better: backtracking with constraint propagation
    # For each tree vertex, the available hosts are those not used and not connected
    # to already-mapped neighbors in a way that creates edge conflicts.
    
    # Actually, the constraint is: for each edge (u,v) in tree, (mapping[u], mapping[v]) must not be in used_edges
    # And mapping must be injective.
    
    # Let's use a simple backtracking with good variable ordering
    
    # Order tree vertices: start from start_leaf, then BFS order
    order = []
    visited = [False] * k
    queue = [start_leaf]
    visited[start_leaf] = True
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in tree_adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    
    # Now backtracking: assign hosts to vertices in order
    # For vertex at position i in order, try all available hosts
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == k:
            return mapping[:]
        
        u = order[idx]
        # Try all possible hosts for u
        for h in range(n):
            if h in used_hosts:
                continue
            
            # Check if assigning h to u creates any edge conflict with already-mapped neighbors
            conflict = False
            for v in tree_adj[u]:
                if mapping[v] != -1:
                    edge = (min(h, mapping[v]), max(h, mapping[v]))
                    if edge in used_edges:
                        conflict = True
                        break
            
            if conflict:
                continue
            
            mapping[u] = h
            used_hosts.add(h)
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            mapping[u] = -1
            used_hosts.remove(h)
        
        return None
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    trees = _generate_trees(20, 611)
    n = 20
    
    # Start with no used edges
    used_edges = set()
    embeddings = [None] * 20
    
    # Process trees in order of increasing size (smaller trees first might be easier?)
    # Or process in original order? Let's try original order first.
    # Actually, let's try processing smaller trees first to fill in the easy ones,
    # then tackle the larger ones.
    
    # Actually, the largest tree (T20) has 19 edges, which is a spanning tree.
    # It's probably the hardest to embed. Let's do it first? Or last?
    # Let's try original order: T1, T2, ..., T20
    
    for k in range(1, 21):
        if time.time() > deadline:
            break
        
        tree_edges = trees[k-1]
        
        if not tree_edges:
            embeddings[k-1] = []
            continue
        
        # Find an embedding
        embedding = _find_embedding(tree_edges, n, used_edges, deadline)
        
        if embedding is None:
            # Try with different randomization? Or just skip?
            # For now, let's try a few random restarts
            found = False
            for attempt in range(10):
                if time.time() > deadline:
                    break
                # Shuffle the order of trying hosts
                # This is a simple heuristic; a better approach would be to
                # randomize the starting vertex or the order of host assignment
                embedding = _find_embedding(tree_edges, n, used_edges, deadline)
                if embedding is not None:
                    found = True
                    break
            
            if not found:
                # Can't embed this tree; mark as None and continue
                embeddings[k-1] = None
                continue
        
        # Extract the mapping: embedding[i] is the host for tree vertex i
        # But we need to return a list where position i is the image of vertex i
        # The embedding from _find_embedding is already in that format
        
        # Add the edges to used_edges
        for u, v in tree_edges:
            edge = (min(embedding[u], embedding[v]), max(embedding[u], embedding[v]))
            used_edges.add(edge)
        
        embeddings[k-1] = [int(embedding[i]) for i in range(len(tree_edges) + 1)]
    
    # Ensure all entries are either None or lists of ints
    result = []
    for i in range(20):
        if embeddings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in embeddings[i]])
    
    return result
