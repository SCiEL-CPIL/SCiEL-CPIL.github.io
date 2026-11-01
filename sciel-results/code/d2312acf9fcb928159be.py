import time
import sys
from collections import deque
import random

def _bfs_girth(adj):
    """Compute the girth of an undirected graph via BFS from each vertex."""
    n = len(adj)
    best = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < best:
                        best = cycle
        if best <= 2:
            break
    return best if best < float('inf') else 0

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _construct_moores_tree_cage(k, g):
    """
    Construct a k-regular graph of girth at least g using a Moore tree completion.
    This works well for small g and k where the Moore tree depth is small.
    """
    if k != 4 or g != 10:
        return None
        
    # Depth d such that 1 + k + k(k-1) + ... + k(k-1)^d >= Moore Bound
    # For k=4, g=10, Moore bound is 161.
    # Depth 4 gives 161 vertices. But we showed depth 4 cannot be completed.
    # Depth 5 gives 485 vertices.
    
    depth = 5
    n = 1
    for i in range(1, depth + 1):
        n += k * (k - 1) ** i
        
    # Build the tree
    # Node 0 is root.
    # We store parent and children to compute distances later if needed, 
    # but for matching we just need to know which nodes are in which layer and their distances.
    
    # Actually, we can just build the adjacency list of the tree first.
    adj = [[] for _ in range(n)]
    layer_of = [0] * n
    layer_of[0] = 0
    
    # BFS to assign layers and edges
    # But we need to generate the tree structure specifically.
    # Root 0.
    # Layer 1: 4 nodes.
    # Layer 2: 12 nodes.
    # ...
    
    # Let's explicitly build the tree edges.
    current_layer = [0]
    next_layer = []
    edges = []
    
    for d in range(1, depth + 1):
        for u in current_layer:
            # Each node in layer d-1 (except root) has k-1 children.
            # Root has k children.
            num_children = k if d == 1 else k - 1
            for i in range(num_children):
                v = len(next_layer) + len(adj) # This logic is flawed, let's just use a counter
                pass
        
    # Simpler: Just generate nodes and connect them in a tree structure.
    # Node IDs:
    # Layer 0: 0
    # Layer 1: 1..4
    # Layer 2: 5..16
    # Layer 3: 17..52
    # Layer 4: 53..160
    # Layer 5: 161..484
    
    # Let's map node ID to layer.
    node_layer = {}
    current_id = 0
    node_layer[current_id] = 0
    layers = [[current_id]]
    current_id += 1
    
    for d in range(1, depth + 1):
        layer_nodes = []
        for u in layers[d-1]:
            num_children = k if d == 1 else k - 1
            for _ in range(num_children):
                v = current_id
                current_id += 1
                layer_nodes.append(v)
                node_layer[v] = d
                # Add edge u-v
                edges.append((u, v))
        layers.append(layer_nodes)
    
    n = current_id
    
    # Initialize adj
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Now, we have a tree.
    # Leaves are in Layer 5.
    # Each leaf has degree 1. It needs 3 more edges.
    # Internal nodes (Layer 0-4) have degree:
    # Layer 0: 4 (full)
    # Layer 1: 1 parent + 3 children = 4 (full)
    # Layer 2: 1 parent + 3 children = 4 (full)
    # Layer 3: 1 parent + 3 children = 4 (full)
    # Layer 4: 1 parent + 3 children = 4 (full)
    # So only Layer 5 nodes have stubs.
    
    leaves = layers[depth]
    num_leaves = len(leaves)
    # Each leaf needs 3 more edges.
    # Total stubs = 3 * num_leaves.
    # We need to pair these stubs.
    
    # To avoid short cycles, we can only connect two leaves u, v if dist(u, v) >= g - 1.
    # g = 10, so dist(u, v) >= 9.
    
    # Compute distance between all pairs of leaves?
    # num_leaves = 324.
    # Pairs = 324 * 323 / 2 ~ 52000.
    # BFS from each leaf to compute distances to other leaves is 324 * 485 ~ 150k ops. Fast.
    
    # However, we don't need all pairs. We just need to find a perfect matching in the "stubs" graph.
    # This is equivalent to finding a 3-regular graph on the set of leaves, where edges exist only if dist(u,v) >= 9.
    # Wait, each leaf has 3 stubs. So we need to decompose the graph of allowed leaf-leaf connections into 3 perfect matchings?
    # Or just find a 3-regular subgraph on the leaves with min degree 3 and max degree 3?
    # Yes, we need a 3-regular graph on the 324 leaves, using only edges with dist >= 9.
    
    # Let's build the allowed graph on leaves.
    # adj_leaf[u] = list of leaves v such that dist(u,v) >= 9.
    
    # First, compute distances from each leaf to all other leaves.
    # We can do BFS from each leaf.
    
    leaf_indices = {leaf: i for i, leaf in enumerate(leaves)}
    L = len(leaves)
    
    # dist_matrix[i][j] = distance between leaves[i] and leaves[j]
    # We only need to check if dist >= 9.
    
    allowed_adj = [[] for _ in range(L)]
    
    for i, leaf_u in enumerate(leaves):
        # BFS from leaf_u
        dist = [-1] * n
        dist[leaf_u] = 0
        q = deque([leaf_u])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                    
        for j, leaf_v in enumerate(leaves):
            if i != j and dist[leaf_v] >= 9:
                allowed_adj[i].append(j)
                
    # Now we need to find a 3-regular subgraph in allowed_adj.
    # This is a random 3-regular graph generation problem with constraints.
    # We can use a greedy approach:
    # 1. Shuffle allowed edges.
    # 2. Try to assign 3 edges to each node.
    
    # Since the graph is dense (dist >= 9 is a large portion of the tree),
    # a random greedy matching should work.
    
    # Let's try to find 3 disjoint perfect matchings.
    # Or just find a 3-regular graph directly.
    
    # Greedy:
    # Maintain degree of each node in the new graph.
    # While there are nodes with degree < 3:
    #   Pick a random node u with degree < 3.
    #   Pick a random neighbor v from allowed_adj[u] such that degree[v] < 3 and (u,v) not used.
    #   Add edge (u,v).
    
    # This might get stuck.
    # Better: Find 3 perfect matchings sequentially.
    
    used_edges = set()
    new_edges = []
    
    for _ in range(3):
        # Find a perfect matching in the remaining allowed graph
        # where each node has capacity 1.
        # This is a standard bipartite matching if we split nodes?
        # No, it's a general graph matching.
        # We can use a simple randomized greedy matching.
        
        match = [-1] * L
        # Shuffle nodes
        nodes = list(range(L))
        random.shuffle(nodes)
        
        for u in nodes:
            if match[u] != -1:
                continue
            # Find a free neighbor v
            neighbors = allowed_adj[u]
            random.shuffle(neighbors)
            found = False
            for v in neighbors:
                if match[v] == -1 and (u, v) not in used_edges and (v, u) not in used_edges:
                    # Check if edge is allowed (it is, from allowed_adj)
                    match[u] = v
                    match[v] = u
                    used_edges.add((u, v))
                    used_edges.add((v, u))
                    new_edges.append((leaves[u], leaves[v]))
                    found = True
                    break
            if not found:
                # Backtrack? Or restart?
                # For now, if we fail, return None
                return None
                
    # If we succeeded, we have 3 * L / 2 = 3 * 162 = 486 edges?
    # No, each matching has L/2 = 162 edges.
    # 3 matchings = 486 edges.
    # Total edges in final graph = Tree edges (484) + 486 = 970.
    # Degree sum = 2 * 970 = 1940.
    # n * k = 485 * 4 = 1940.
    # Matches.
    
    # Add new edges to adj
    for u, v in new_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Verify girth
    girth = _bfs_girth(adj)
    if girth >= g:
        return n, edges + new_edges
    else:
        return None

def construct(k, g):
    """Construct a k-regular graph of girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if k == 4 and g == 10:
        # Try the Moore tree completion
        try:
            result = _construct_moores_tree_cage(k, g)
            if result:
                n, edges = result
                return n, edges
        except Exception:
            pass
            
        # Fallback to LUW
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        # General case: use LUW construction
        n, edges = _luw_construct(k, g)
        return n, edges
