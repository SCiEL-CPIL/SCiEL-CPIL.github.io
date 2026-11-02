"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using random greedy expansion
    of the k-regular tree of radius floor((g-1)/2).
    
    Strategy:
    1. Build the k-regular tree T of radius r = floor((g-1)/2) from a root.
       This tree has girth >= g (actually infinite, but we'll close it).
    2. The leaves of this tree need to be connected to achieve k-regularity.
       Each leaf currently has degree 1 (except root which has degree k).
    3. Iteratively add edges between pairs of leaves (or vertices with degree < k)
       such that the distance in the current graph is at least g.
    4. Use a greedy randomized approach with backtracking if stuck.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # For the specific task: k=5, g=9
    # The 5-regular tree of radius 4 has:
    # Level 0: 1 vertex (root)
    # Level 1: 5 vertices
    # Level 2: 5*4 = 20 vertices
    # Level 3: 20*4 = 80 vertices
    # Level 4: 80*4 = 320 vertices
    # Total: 1 + 5 + 20 + 80 + 320 = 426 vertices
    
    # But we need a 5-regular graph, so the tree alone isn't enough.
    # We need to add edges between leaves to make it 5-regular.
    
    # Let's use a different approach: start with a random 5-regular graph 
    # and try to fix short cycles, OR use the tree-based construction.
    
    # Actually, let me implement the tree-based greedy approach properly.
    # The key insight: start with the girth-g tree, then add edges between
    # vertices that are far apart in the tree to maintain girth >= g.
    
    r = (g - 1) // 2  # radius of the tree
    
    # Build the k-regular tree of radius r
    # Vertex 0 is the root
    # Each vertex at level < r has k children (except root which has k)
    # Vertices at level r are leaves
    
    # Count vertices in the tree
    # Level 0: 1
    # Level i (i>=1): (k-1)^(i-1) * k for i < r, but actually:
    # Root has k neighbors
    # Each non-root vertex at level < r has (k-1) new neighbors
    # So level i has: k * (k-1)^(i-1) for i >= 1
    
    tree_sizes = [1]
    for i in range(1, r + 1):
        tree_sizes.append(k * (k - 1) ** (i - 1))
    
    n_tree = sum(tree_sizes)
    
    # Build the tree
    # Assign vertex IDs: root = 0
    # Level 1: 1 to k
    # Level 2: k+1 to k + k*(k-1)
    # etc.
    
    tree_edges = []
    parent = [0] * n_tree
    level = [0] * n_tree
    children = [[] for _ in range(n_tree)]
    
    # Assign levels and parents
    vid = 0
    level[0] = 0
    parent[0] = -1
    
    for lvl in range(1, r + 1):
        start_vid = sum(tree_sizes[:lvl])
        for i in range(tree_sizes[lvl]):
            v = start_vid + i
            level[v] = lvl
            # Parent: for level 1, parent is 0
            # For level > 1, each parent at level lvl-1 has (k-1) children
            # Parent index within its level:
            if lvl == 1:
                p = 0
            else:
                prev_start = sum(tree_sizes[:lvl-1])
                # Each vertex at level lvl-1 has (k-1) children
                # Vertex i at level lvl has parent at position i // (k-1) at level lvl-1
                p_idx = i // (k - 1)
                p = prev_start + p_idx
            parent[v] = p
            children[p].append(v)
            tree_edges.append((p, v))
    
    # Now we have a tree with n_tree vertices.
    # Degrees in the tree:
    # Root: k
    # Internal non-root: 1 (parent) + (k-1) (children) = k
    # Leaves: 1 (parent)
    
    # We need to add edges to make all vertices have degree k.
    # Leaves need (k-1) more edges each.
    # The number of leaves is tree_sizes[r] = k * (k-1)^(r-1)
    
    n_leaves = tree_sizes[r]
    edges_to_add = n_leaves * (k - 1) // 2  # each edge adds 2 to total degree
    
    # For k=5, g=9, r=4:
    # n_leaves = 5 * 4^3 = 320
    # edges_to_add = 320 * 4 / 2 = 640
    
    # Total edges will be: tree_edges (425) + 640 = 1065
    # For a 5-regular graph on 426 vertices: 426*5/2 = 1065. Checks out!
    
    # Now we need to add 640 edges between leaves such that:
    # 1. No two leaves are connected if their distance in the tree is < g
    # 2. The resulting graph has girth >= g
    # 3. Each leaf gets exactly (k-1) new edges
    
    # Distance between two leaves in the tree:
    # If leaves are in different branches from the root, distance = 2*r = 8 for r=4
    # Wait, that's less than g=9! So we can't just connect arbitrary leaves.
    
    # For g=9, r=4, the distance between any two leaves in the tree is at most 8
    # (going up to LCA and back down). So we CANNOT form a 5-regular graph
    # on just the 426 vertices of the tree with girth >= 9.
    
    # This means we need MORE vertices. The tree of radius 4 doesn't work directly.
    
    # Alternative approach: Use a larger structure.
    # Let's try a different strategy: start with a random 5-regular graph 
    # and use local search to eliminate short cycles.
    
    # Or: use the fact that we need at least 428 vertices (lower bound).
    # Let's try to construct on ~500-1000 vertices using random greedy.
    
    # New strategy: Random greedy construction
    # 1. Start with n vertices
    # 2. Build a 5-regular graph randomly
    # 3. Find short cycles and try to fix them by edge swaps
    # 4. Repeat until girth >= 9
    
    # Let's try n = 500 first, then increase if needed
    
    best_result = None
    
    for n_attempt in [500, 600, 700, 800, 1000, 1200]:
        if time.time() > deadline:
            break
        
        # Try to construct a 5-regular graph on n_attempt vertices with girth >= 9
        result = _try_construct(k, g, n_attempt, deadline)
        if result is not None:
            n, edges = result
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
            if n <= 600:  # good enough, stop
                break
    
    if best_result is None:
        # Fallback: return the tree-based construction (may not be valid but better than nothing)
        # Actually, let's just return a valid 5-regular graph even if girth < 9
        # No, that would score 0. Let's try one more time with more vertices.
        result = _try_construct(k, g, 1500, deadline)
        if result is not None:
            best_result = result
    
    if best_result is None:
        # Last resort: return the LZW graph from the original code
        return _lzw_construct(k, g)
    
    return best_result


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


def _try_construct(k, g, n, deadline):
    """
    Try to construct a k-regular graph on n vertices with girth >= g
    using random greedy with local search.
    """
    # First, generate a random k-regular graph
    # Then check girth and try to fix short cycles
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Generate random k-regular graph
        edges = _random_regular_graph(k, n)
        if edges is None:
            continue
        
        # Check girth
        girth = _compute_girth_fast(n, edges, g)
        
        if girth >= g:
            return (n, edges)
        
        # Try to fix short cycles
        edges = _fix_short_cycles(n, edges, k, g, deadline)
        if edges is None:
            continue
        
        girth = _compute_girth_fast(n, edges, g)
        if girth >= g:
            return (n, edges)
    
    return None


def _random_regular_graph(k, n):
    """Generate a random k-regular graph on n vertices"""
    if n * k % 2 != 0:
        return None
    
    # Use the configuration model
    # Create 2n stubs: each vertex gets k stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    # Randomly pair stubs
    random.shuffle(stubs)
    
    edges = []
    seen = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None  # loop, retry
        key = (min(u, v), max(u, v))
        if key in seen:
            return None  # duplicate edge, retry
        seen.add(key)
        edges.append((u, v))
    
    return edges


def _compute_girth(n, edges):
    """Compute the girth of a graph using BFS from each vertex"""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        # BFS to find shortest cycle containing start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            node = queue[head]
            head += 1
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    # Found a cycle
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 3:
                        return min_girth
        
        if min_girth <= 3:
            return min_girth
    
    return min_girth if min_girth != float('inf') else n


def _fix_short_cycles(n, edges, k, g, deadline):
    """
    Try to fix short cycles by edge swaps.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all short cycles
    short_cycles = _find_short_cycles(n, adj, g)
    
    if not short_cycles:
        return edges
    
    # Try to fix by removing an edge from a short cycle and adding a new edge
    for _ in range(100):
        if time.time() > deadline:
            return None
        
        cycle = random.choice(short_cycles)
        # Remove one edge from the cycle
        edge_idx = random.randint(0, len(cycle) - 1)
        u, v = cycle[edge_idx]
        
        # Find new edges to add
        new_edge = None
        for _ in range(50):
            # Pick random vertices that are not adjacent
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b or b in adj[a]:
                continue
            # Check if adding (a,b) creates a short cycle
            if _would_create_short_cycle(n, adj, a, b, g):
                continue
            new_edge = (a, b)
            break
        
        if new_edge is None:
            continue
        
        # Perform the swap: remove (u,v), add new_edge
        # But we need to maintain k-regularity, so we also need to adjust degrees
        # This is tricky. Let's try a different approach.
        
        # Actually, for a k-regular graph, we can't just swap edges easily.
        # Let's try a 2-switch: find two edges (u,v) and (a,b) where u!=a, u!=b, v!=a, v!=b
        # and replace them with (u,a) and (v,b) or (u,b) and (v,a)
        
        # Find another edge to swap with
        for _ in range(50):
            c, d = random.choice(edges)
            if c == u or c == v or d == u or d == v:
                continue
            if d in adj[c]:  # should always be true
                continue
            # Try swap: (u,v) and (c,d) -> (u,c) and (v,d)
            if _would_create_short_cycle(n, adj, u, c, g) or _would_create_short_cycle(n, adj, v, d, g):
                continue
            # Perform the swap
            edges.remove((u, v))
            edges.remove((c, d))
            edges.append((u, c))
            edges.append((v, d))
            # Update adj
            adj[u].remove(v)
            adj[v].remove(u)
            adj[c].remove(d)
            adj[d].remove(c)
            adj[u].append(c)
            adj[c].append(u)
            adj[v].append(d)
            adj[d].append(v)
            break
    
    # Re-check girth
    girth = _compute_girth(n, edges)
    if girth >= g:
        return edges
    
    return None


def _find_short_cycles(n, adj, max_len):
    """Find all cycles of length < max_len"""
    cycles = []
    for start in range(n):
        # DFS to find cycles of length < max_len
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) >= max_len:
                continue
            for nb in adj[node]:
                if nb == path[0] and len(path) >= 3:
                    cycles.append(list(zip(path, path[1:] + [path[0]])))
                elif nb not in path:
                    stack.append((nb, path + [nb]))
    return cycles


def _would_create_short_cycle(n, adj, u, v, g):
    """Check if adding edge (u,v) would create a cycle of length < g"""
    # BFS from u to v, find shortest path
    # If shortest path length < g-1, then adding (u,v) creates a cycle of length < g
    dist = [-1] * n
    dist[u] = 0
    queue = [u]
    
    for node in queue:
        if dist[node] >= g - 1:
            break
        for nb in adj[node]:
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                queue.append(nb)
                if nb == v:
                    return dist[v] < g - 1
    
    return dist[v] < g - 1
# EVOLVE-BLOCK-END
