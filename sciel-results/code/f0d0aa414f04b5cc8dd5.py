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
    if k != 4 or g != 10:
        return None
        
    depth = 5
    # Build the (k-1)-regular tree of depth 5 rooted at 0
    edges = []
    layers = [[0]]
    current_id = 1
    parent = [0]  # parent[0] = 0 (root)
    
    for d in range(1, depth + 1):
        layer_nodes = []
        for u in layers[d-1]:
            num_children = k if d == 1 else k - 1
            for _ in range(num_children):
                v = current_id
                current_id += 1
                layer_nodes.append(v)
                parent.append(u)
                edges.append((u, v))
        layers.append(layer_nodes)
        
    n = current_id
    leaves = layers[depth]
    L = len(leaves)
    
    # Map each leaf to its branch (layer-1 ancestor: node 1,2,3,4)
    leaf_branch = {}
    for leaf in leaves:
        cur = leaf
        while parent[cur] != 0:
            cur = parent[cur]
        leaf_branch[leaf] = cur
    
    # Group leaves by branch
    branch_leaves = {b: [] for b in range(1, 5)}
    for leaf in leaves:
        branch_leaves[leaf_branch[leaf]].append(leaf)
    
    # Each branch has 81 leaves. We need a 3-regular graph on all 324 leaves
    # with no edges within the same branch.
    # 
    # Strategy: Use a 1-factorization approach.
    # For a 3-regular graph with no intra-branch edges, we can decompose into
    # 3 perfect matchings, each of which is a perfect matching with no
    # intra-branch edges.
    #
    # Since 324 = 4 * 81, and we need matchings between different branches,
    # we can use the structure: for each "slot" s in 0..80, we match
    # branch[0][s] with branch[1][s], branch[2][s] with branch[3][s], etc.
    # But we need 3 such matchings, and each vertex appears in exactly one
    # matching per "round".
    #
    # Better approach: Think of it as a 3-regular bipartite-like structure.
    # Actually, let's use a cleaner algebraic construction.
    #
    # Label leaves in branch b (b=0,1,2,3) as (b, i) where i = 0..80.
    # We want 3 perfect matchings M1, M2, M3 such that:
    #   - Each Mi pairs vertices from different branches
    #   - Every vertex appears in exactly one Mi
    #
    # Use cyclic shifts. For matching m (m=0,1,2):
    #   Pair (b, i) with (b', i') where b' = (b + m + 1) % 4, i' = (i + m * 27) % 81
    # This ensures no intra-branch edges since b' != b.
    # And it's a perfect matching since the mapping is bijective.
    
    new_edges = []
    edge_set = set()
    
    for m in range(3):
        # For this matching, pair each vertex with its partner
        # We only add each edge once (from the lower-indexed vertex's perspective)
        for b in range(4):
            for i in range(81):
                u = branch_leaves[b+1][i]  # branch_leaves is keyed by 1,2,3,4
                b_prime = (b + m + 1) % 4
                i_prime = (i + m * 27) % 81
                v = branch_leaves[b_prime + 1][i_prime]
                
                if u < v:
                    edge_key = (u, v)
                else:
                    edge_key = (v, u)
                
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    new_edges.append(edge_key)
    
    # Verify: each leaf should have degree exactly 3
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in new_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for leaf in leaves:
        if len(adj[leaf]) != 3:
            return None
    
    # Check girth
    girth = _bfs_girth(adj)
    if girth >= g:
        return n, edges + new_edges
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    if k == 4 and g == 10:
        try:
            if time.time() < deadline:
                result = _construct_moores_tree_cage(k, g)
                if result:
                    n, edges = result
                    return n, edges
        except Exception:
            pass
            
        if time.time() >= deadline:
            # Fallback: return LUW quickly
            try:
                n, edges = _luw_construct(k, g)
                return n, edges
            except Exception:
                pass
        
        n, edges = _luw_construct(k, g)
        return n, edges
    else:
        n, edges = _luw_construct(k, g)
        return n, edges
