import time as _time
import math
import random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """
    Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y:(x+y)%q
        sub = lambda x,y:(x-y)%q
        mul = lambda x,y:x*y%q
        
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
            
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _greedy_tree_extension(k, g, target_n, seed=42, time_limit=60.0):
    """
    Randomized greedy tree-extension algorithm.
    Starts from a random tree of depth ~g/2, then greedily adds edges
    to close the graph while maintaining girth >= g.
    """
    start_time = _time.time()
    random.seed(seed)
    
    # Moore bound estimate for minimum n
    # For k=3, g=16: Moore bound is 2 * (1 + 2 + 4 + ... + 2^7) = 2*(2^8-1) = 510
    # But we need to be practical. Let's aim for n around 1000-2000 initially.
    
    # Build a random tree with branching factor k-1=2 (binary tree) to depth d
    # Number of vertices in full binary tree of depth d: 2^(d+1) - 1
    # For girth g, we need depth at least g/2 - 1 to avoid small cycles
    depth = max(3, g // 2 - 1)
    tree_n = (1 << (depth + 1)) - 1  # 2^(depth+1) - 1
    
    # If tree_n is too large, reduce depth
    while tree_n > target_n and depth > 2:
        depth -= 1
        tree_n = (1 << (depth + 1)) - 1
    
    # Build the tree
    # Vertex 0 is root. Children of v at depth < depth: 2v+1, 2v+2
    # This gives a complete binary tree structure
    
    # We'll build a partial tree and then add edges greedily
    # Start with a random permutation of a large tree, then add edges
    
    # Alternative: start with n vertices in a random order, build a spanning tree,
    # then add edges while checking girth
    
    n = target_n
    adj = [set() for _ in range(n)]
    edges = []
    
    # Build a random spanning tree
    # Use a random BFS/DFS order
    order = list(range(n))
    random.shuffle(order)
    
    # Build tree: connect each vertex (except root) to a random earlier vertex
    # This creates a random tree
    parent = [-1] * n
    for i in range(1, n):
        # Pick a random parent from earlier vertices
        p = random.randint(0, i - 1)
        parent[i] = p
        adj[i].add(p)
        adj[p].add(i)
        edges.append((p, i))
    
    # Now we have a tree with n-1 edges. Each vertex has degree 1 (except root) or more.
    # We need to add edges to make it k-regular (k=3).
    # Current degrees: root has 0, others have 1 (plus their children)
    
    # Compute current degrees
    deg = [len(adj[i]) for i in range(n)]
    
    # Vertices that need more edges (degree < k)
    # In a tree, leaves have degree 1, internal nodes have higher degree
    # We need to add edges to make all degrees exactly k=3
    
    # Strategy: repeatedly pick two vertices with degree < k, not adjacent,
    # and add an edge between them if it doesn't create a cycle shorter than g
    
    # Precompute distance from each vertex to all others up to depth g-1
    # This is expensive for large n, so we'll do it on the fly
    
    def _shortest_path_exists(u, v, adj, max_dist):
        """Check if there's a path from u to v of length <= max_dist, not using direct edge if exists."""
        if u == v:
            return 0
        visited = {u: 0}
        dq = deque([u])
        while dq:
            cur, d = dq.popleft()
            if d >= max_dist:
                continue
            for nb in adj[cur]:
                if nb == v:
                    return d + 1
                if nb not in visited:
                    visited[nb] = d + 1
                    dq.append((nb, d + 1))
        return None
    
    def _would_create_short_cycle(u, v, adj, g):
        """Check if adding edge (u,v) would create a cycle shorter than g."""
        # A cycle through (u,v) has length = dist(u,v) + 1
        # We need dist(u,v) + 1 >= g, i.e., dist(u,v) >= g-1
        # But we also need to check all paths, not just shortest
        # Actually, if the shortest path is >= g-1, then all paths are >= g-1
        # So we just need shortest_path >= g-1
        dist = _shortest_path_exists(u, v, adj, g - 2)
        if dist is None:
            # No path within g-2, so any cycle would be >= g
            return False
        # If dist < g-1, then cycle length = dist + 1 < g
        return dist < g - 1
    
    # Greedily add edges
    max_iterations = 100000
    for _ in range(max_iterations):
        if _time.time() - start_time > time_limit:
            break
        
        # Find vertices with degree < k
        under_deg = [i for i in range(n) if deg[i] < k]
        if not under_deg:
            break
        
        # Try to add an edge between two under-degree vertices
        added = False
        random.shuffle(under_deg)
        
        for i in range(len(under_deg)):
            for j in range(i + 1, len(under_deg)):
                u, v = under_deg[i], under_deg[j]
                if u in adj[v]:
                    continue  # Already adjacent
                if _would_create_short_cycle(u, v, adj, g):
                    continue
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
                added = True
                break
            if added:
                break
        
        if not added:
            # No valid edge could be added
            # Try a different strategy: remove some tree edges and restructure
            # For now, break and return what we have
            break
    
    # Check if we achieved k-regular
    if all(deg[i] == k for i in range(n)):
        # Verify girth
        adj_list = [list(s) for s in adj]
        if _verify_girth_fast(adj_list, n, g):
            return n, edges
    
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140
    
    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, g=16
    best_n = float('inf')
    best_result = None
    
    # Try greedy tree-extension with various target sizes
    # Moore bound for (3,16) is 510, but practical minimum is higher
    # Try sizes from 600 to 1500
    target_sizes = [600, 800, 1000, 1200, 1400, 1600]
    
    for target_n in target_sizes:
        if _time.time() > deadline - 30:
            break
        if target_n >= best_n:
            break
        
        # Try a few seeds
        for seed in [42, 123, 456]:
            if _time.time() > deadline - 10:
                break
            try:
                result = _greedy_tree_extension(k, g, target_n, seed=seed, time_limit=15.0)
                if result is not None:
                    n, edges = result
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break  # Found valid for this target_n, try next (smaller) target
            except Exception:
                pass
    
    # Fallback to LUW if greedy didn't find anything good
    if best_result is None or best_n > 2000:
        for d in [7, 8]:
            if _time.time() > deadline - 20:
                break
            try:
                n_luw, edges_luw = _luw_construct(3, g, d_override=d)
                adj = [set() for _ in range(n_luw)]
                for u, v in edges_luw:
                    adj[u].add(v)
                    adj[v].add(u)
                adj_list = [list(s) for s in adj]
                if _verify_girth_fast(adj_list, n_luw, g):
                    if n_luw < best_n:
                        best_n = n_luw
                        best_result = (n_luw, edges_luw)
                    break
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    n_luw, edges_luw = _luw_construct(3, g, d_override=10)
    return n_luw, edges_luw
