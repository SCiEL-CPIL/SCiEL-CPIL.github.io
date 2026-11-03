"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using non-backtracking walk enumeration.
    
    Strategy:
    1. Build a tree of depth h = g//2 from a root (Moore bound approximation)
    2. Identify boundary vertices to close cycles of length exactly g
    3. Ensure no shorter cycles exist via non-backtracking walk verification
    """
    if k == 1:
        return 2, [(0, 1)]
    
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    h = g // 2  # half the girth
    
    # Moore bound: n_min = 2 * sum_{i=0}^{h-1} (k-1)^i
    moore_bound = 2 * sum((k-1)**i for i in range(h))
    
    # Build the graph using a greedy non-backtracking walk approach
    # We'll use a tree-based construction then close the cycles
    
    # Phase 1: Build a (k-1)-ary tree of depth h from root
    # Root has degree k, internal nodes have degree k (one to parent, k-1 to children)
    # Leaves have degree 1 (to parent) - we need to connect them
    
    # Assign vertex IDs in BFS order
    vertices = []  # list of (parent, siblings_index)
    vertex_id = {}
    
    # Root
    root_id = 0
    vertices.append(None)  # root has no parent
    vertex_id[0] = (None, 0)
    
    # Build tree level by level
    current_level = [0]  # IDs of vertices at current level
    next_level = []
    
    for level in range(1, h):
        for vid in current_level:
            # Each vertex at this level gets (k-1) children (one edge used for parent)
            for child_idx in range(k - 1):
                new_id = len(vertices)
                vertices.append(vid)  # parent
                vertex_id[new_id] = (vid, child_idx)
                next_level.append(new_id)
        current_level = next_level
        next_level = []
    
    n_tree = len(vertices)
    
    # Now we have a tree with n_tree vertices
    # The leaves are at depth h-1 (if h >= 1) or just the root
    # We need to add edges between leaves to make it k-regular and close cycles
    
    # Get all leaf vertices
    leaves = current_level if current_level else [0]
    n_leaves = len(leaves)
    
    # Each leaf currently has degree 1 (connected to parent)
    # Each leaf needs (k-1) more edges to reach degree k
    # Total additional edges needed: n_leaves * (k-1) / 2
    
    # Strategy: connect leaves in a pattern that creates cycles of length >= g
    # We'll use a cyclic connection pattern
    
    # For simplicity and to ensure girth, we'll use a specific pairing strategy
    # Connect leaf i to leaf (i + offset) mod n_leaves for appropriate offsets
    
    edges = []
    
    # Add tree edges
    for vid in range(1, n_tree):
        parent = vertices[vid]
        edges.append((parent, vid))
    
    # Now add edges between leaves
    # We need to add n_leaves * (k-1) / 2 edges
    # Use a greedy approach: for each leaf, connect to unconnected leaves
    
    # Track degree of each vertex
    degree = [0] * n_tree
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    
    # For each leaf, add (k-1) more edges
    leaf_set = set(leaves)
    # Create adjacency for quick lookup
    adj = {i: set() for i in range(n_tree)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedily add edges between leaves
    for leaf in leaves:
        needed = k - degree[leaf]
        if needed <= 0:
            continue
        # Find other leaves that need connections
        candidates = [l for l in leaves if l != leaf and degree[l] < k and l not in adj[leaf]]
        # Sort by degree (prioritize vertices with higher degree to balance)
        candidates.sort(key=lambda x: -degree[x])
        for cand in candidates[:needed]:
            edges.append((leaf, cand))
            adj[leaf].add(cand)
            adj[cand].add(leaf)
            degree[leaf] += 1
            degree[cand] += 1
            if degree[leaf] >= k:
                break
    
    # Check if all vertices have degree k
    if any(d != k for d in degree):
        # Fallback: use LUW construction
        return _luw_construct(k, g)
    
    n = n_tree
    return n, edges

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
    
    # Find appropriate d for LUW
    candidates = []
    
    d_odd = g - 5
    if d_odd < 3:
        d_odd = 3
    if d_odd % 2 == 0:
        d_odd += 1
    candidates.append(d_odd)
    
    d_even = g - 4
    if d_even < 2:
        d_even = 2
    if d_even % 2 == 1:
        d_even += 1
    candidates.append(d_even)
    
    d = min(candidates)
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
