"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. Try tree-expansion construction (build from cage tree).
    2. Fallback to LZW Algebraic Construction.
    """
    deadline = time.time() + 110  # Safety margin
    
    # Try tree-expansion construction
    result = _tree_expansion_construct(k, g, deadline)
    if result is not None:
        return result
        
    # Fallback to LZW
    return _lzw_construct(k, g)

def _tree_expansion_construct(k, g, deadline):
    """
    Construct a k-regular graph with girth >= g by expanding the cage tree.
    
    The cage tree is the unique k-regular tree of radius floor((g-1)/2).
    For k=5, g=9: radius = 4, so the tree has 1 + 5 + 25 + 125 + 625 = 781 vertices.
    
    We start with this tree and add edges between leaf vertices at distance >= g
    until the graph is k-regular.
    """
    radius = (g - 1) // 2
    
    # Build the cage tree
    # Level 0: root (1 vertex)
    # Level 1: k vertices
    # Level 2: k^2 vertices
    # ...
    # Level radius: k^radius vertices
    
    # Count vertices in tree
    tree_size = 1
    for i in range(1, radius + 1):
        tree_size += k ** i
    
    # For k=5, radius=4: 1 + 5 + 25 + 125 + 625 = 781
    # Target n should be at least tree_size, but we may need more vertices
    # Actually, the tree IS the graph structure - we just add edges between leaves
    
    # Build the tree structure
    # Vertex 0 is root
    # Vertices 1..k are level 1
    # etc.
    
    # Let's use a different indexing:
    # Level 0: vertex 0
    # Level 1: vertices 1 to k
    # Level 2: vertices k+1 to k + k^2
    # Level 3: vertices k + k^2 + 1 to k + k^2 + k^3
    # Level 4: vertices k + k^2 + k^3 + 1 to k + k^2 + k^3 + k^4
    
    # Build parent-child relationships
    # For a vertex at level L (L >= 1), its parent is at level L-1
    # Vertex at level L with index j (0-indexed within level) has parent at level L-1 with index j // k
    
    # Let's compute level boundaries
    level_start = [0]
    for i in range(1, radius + 1):
        level_start.append(level_start[-1] + k ** (i - 1))
    level_start.append(level_start[-1] + k ** radius)
    
    # level_start[i] is the start index of level i
    # Level i has vertices from level_start[i] to level_start[i+1] - 1
    
    # Build tree edges
    tree_edges = []
    for level in range(1, radius + 1):
        for j in range(k ** level):
            child_idx = level_start[level] + j
            parent_idx = level_start[level - 1] + (j // k)
            tree_edges.append((parent_idx, child_idx))
    
    # Now we have a tree with tree_size vertices
    # Each vertex at level < radius has degree k (except root which has degree k)
    # Vertices at level radius have degree 1 (only connected to parent)
    
    # We need to add edges between leaf vertices (level radius) to make all degrees = k
    # Each leaf currently has degree 1, needs k-1 more edges
    # Total edges to add: tree_size * k / 2 - len(tree_edges)
    
    target_edges = tree_size * k // 2
    edges_to_add = target_edges - len(tree_edges)
    
    # Leaves are vertices at level radius
    leaves = list(range(level_start[radius], level_start[radius + 1]))
    
    # We need to pair up leaves and add edges between them
    # But we need to ensure distance >= g in the tree
    
    # For now, let's try a simple approach:
    # Shuffle leaves and greedily add edges between pairs that are far apart
    
    # First, compute pairwise distances between all leaves? That's O(n^2) which might be too slow
    # For 625 leaves, that's 625^2 / 2 = ~195K pairs, which is manageable
    
    # Let's compute distances using BFS from each leaf
    # Or better: use the tree structure to compute LCA and distances
    
    # Actually, let's just do it simply:
    # For each pair of leaves, compute their distance in the tree
    # Then greedily add edges between pairs with distance >= g
    
    # Build adjacency for the tree
    adj = [[] for _ in range(tree_size)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Compute distance between two nodes in the tree using LCA
    # Precompute parent and depth for each node
    parent = [-1] * tree_size
    depth = [0] * tree_size
    
    # BFS from root
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if parent[nb] == -1:
                parent[nb] = node
                depth[nb] = depth[node] + 1
                queue.append(nb)
    
    def lca(u, v):
        # Bring to same depth
        while depth[u] > depth[v]:
            u = parent[u]
        while depth[v] > depth[u]:
            v = parent[v]
        # Move up together
        while u != v:
            u = parent[u]
            v = parent[v]
        return u
    
    def tree_dist(u, v):
        w = lca(u, v)
        return depth[u] + depth[v] - 2 * depth[w]
    
    # Now we need to add edges_to_add edges between leaves
    # Each leaf needs (k-1) additional edges
    # Total leaf-edge-ends needed: len(leaves) * (k-1) = 625 * 4 = 2500
    # So we need 2500/2 = 1250 edges between leaves
    
    # Let's create a list of leaf pairs and their distances
    # Then sort by distance and greedily pick pairs with distance >= g
    
    # For efficiency, let's just randomly sample pairs and check distance
    
    # Better approach: for each leaf, find all other leaves at distance >= g
    # Then use a matching algorithm
    
    # Simpler: just try random pairs until we have enough valid edges
    
    current_edges = set(tree_edges)
    added_edges = []
    
    # Track how many more edges each leaf needs
    leaf_degree_needed = {leaf: k - 1 for leaf in leaves}
    
    # Create a list of leaves sorted by remaining degree needed
    # We'll repeatedly pick two leaves that both need edges and are far apart
    
    # For speed, let's precompute for each leaf the list of leaves at distance >= g
    # This is O(n_leaves^2) = O(625^2) = ~390K operations, which is fast
    
    valid_pairs = []  # List of (leaf1, leaf2, distance)
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            u, v = leaves[i], leaves[j]
            d = tree_dist(u, v)
            if d >= g:
                valid_pairs.append((u, v, d))
    
    # Shuffle valid_pairs to break symmetry
    random.shuffle(valid_pairs)
    
    # Now greedily add edges
    # We need to ensure that when we add an edge, both endpoints still need edges
    # And that we don't add the same edge twice
    
    used_edges = set()
    for u, v, d in valid_pairs:
        if len(added_edges) >= edges_to_add:
            break
        if leaf_degree_needed[u] > 0 and leaf_degree_needed[v] > 0:
            if (u, v) not in used_edges and (v, u) not in used_edges:
                added_edges.append((u, v))
                used_edges.add((u, v))
                used_edges.add((v, u))
                leaf_degree_needed[u] -= 1
                leaf_degree_needed[v] -= 1
    
    # Check if we added enough edges
    if len(added_edges) < edges_to_add:
        # Not enough valid pairs. This shouldn't happen for the cage tree,
        # but if it does, return None to fallback
        return None
    
    # Build the final edge list
    all_edges = tree_edges + added_edges
    
    # Verify the graph is k-regular and has girth >= g
    # Quick check: all vertices should have degree k
    deg = [0] * tree_size
    for u, v in all_edges:
        deg[u] += 1
        deg[v] += 1
    
    if any(d != k for d in deg):
        return None
    
    # Verify girth using BFS from each vertex
    # This is O(n * E) which for n=781, E=1952 is ~1.5M ops, fast
    min_girth = float('inf')
    for start in range(tree_size):
        dist = [-1] * tree_size
        parent_bfs = [-1] * tree_size
        dist[start] = 0
        queue = [start]
        
        for node in queue:
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent_bfs[nb] = node
                    queue.append(nb)
                elif nb != parent_bfs[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return None
    
    if min_girth < g:
        return None
    
    return (tree_size, all_edges)

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
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

# EVOLVE-BLOCK-END
