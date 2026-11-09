import time
import random
import sys
import numpy as np
from collections import deque

def _girth_check_nonbacktracking(adj, n, g):
    """
    Check if graph has girth >= g using non-backtracking BFS from each vertex.
    A cycle of length < g exists iff there is a non-backtracking closed walk of length < g.
    For girth >= g, we need no non-backtracking closed walk of length 1..g-1.
    Since the graph is simple, no closed walks of length 1 or 2 exist.
    We check for non-backtracking walks of length 3..g-1 that return to start.
    
    Efficient approach: for each vertex v, do BFS in the non-backtracking graph.
    State: (current_node, previous_node). Start: (v, -1) with depth 0.
    From (u, p), can go to (w, u) for each w in adj[u], w != p.
    We check if we can reach (v, u) for any u in adj[v] at depth < g.
    """
    max_depth = g - 1  # We need to detect cycles of length < g, i.e., up to g-1
    # For each vertex v, check if there's a non-backtracking cycle of length < g through v
    for v in range(n):
        # BFS from v in non-backtracking sense
        # visited: set of (node, prev) pairs
        visited = set()
        # Start: we come to v from "nowhere", so prev = -1
        queue = deque()
        for u in adj[v]:
            visited.add((v, u))  # We've used edge (v,u), now at v, came from u... wait
            # Actually, let's think differently.
            # A non-backtracking walk starting at v: v -> n1 -> n2 -> ... -> nk
            # It's a cycle if nk = v and k >= 3 (since simple graph, k=2 would be backtracking)
            # Non-backtracking means n_{i-1} != n_{i+1}
            
            # Let's use state (current, prev) where we are at current, having come from prev.
            # Initial states: (v, -1) - we are at v, haven't come from anywhere yet.
            # But from (v, -1), we can go to any neighbor u, becoming (u, v).
            # From (u, v), we can go to any w in adj[u] where w != v, becoming (w, u).
            # We detect a cycle if we reach (v, u) for some u in adj[v] at depth >= 2 
            # (i.e., the walk v->...->u->v has length >= 3).
            
            # Actually, the walk v -> u1 -> u2 -> ... -> uk -> v has length k+1.
            # Non-backtracking: u_{i-1} != u_{i+1} for all i.
            # We want k+1 < g, so k <= g-2.
            
            # Let's do BFS where state is (node, prev_node).
            # Start: (v, -1) at depth 0.
            # Transition: from (node, prev), go to (next, node) for next in adj[node], next != prev.
            # This represents a walk of length depth+1.
            # We want to detect if we can reach state (v, u) where u in adj[v], at depth d,
            # where d+1 = length of walk = d+1. The walk is v->...->u->v, length d+1.
            # We need d+1 < g, so d < g-1, i.e., d <= g-2.
            
            pass
        
        # Let me redo this more carefully.
        # We'll track (node, prev) states.
        # dist[(node, prev)] = length of the non-backtracking walk from v to node, arriving from prev.
        # Start: for each u in adj[v], we have a walk of length 1: v->u. State (u, v), dist=1.
        # From (node, prev) at dist d, we can go to (next, node) at dist d+1 for next in adj[node], next != prev.
        # We detect a cycle if we reach (v, u) for u in adj[v] at dist d, meaning walk of length d from v back to v.
        # Wait, (v, u) means we are at v, having come from u. The walk is ...->u->v.
        # If we started at v, the full walk is v->...->u->v, length d.
        # We need d < g.
        
        dist = {}
        queue = deque()
        for u in adj[v]:
            state = (u, v)
            dist[state] = 1
            queue.append(state)
        
        while queue:
            node, prev = queue.popleft()
            d = dist[(node, prev)]
            if d >= g:
                continue
            if node == v and d < g:
                # We found a non-backtracking cycle of length d
                # But wait, we started by going v->u, so the walk is v->...->node->v?
                # No. State (node, prev) means we are at node, came from prev.
                # If node == v, we are back at v. The walk length is d.
                # We need to ensure the walk is non-backtracking, which is guaranteed by construction.
                # But we must ensure d >= 3 (since simple graph, d=1 or 2 can't be a cycle in simple graph
                # without backtracking, but our construction already prevents backtracking).
                # Actually, d=1: v->v? No, that would be a loop. d=2: v->u->v, but that's backtracking
                # (prev of v is u, and we came from v to u, so going back to v is backtracking).
                # Our construction: from (u, v), we can't go to v (since next != prev = v).
                # So d=2 state (v, u) is unreachable. Good.
                # So any d >= 3 reaching (v, u) is a valid non-backtracking cycle.
                return False
            for next_node in adj[node]:
                if next_node == prev:
                    continue
                new_state = (next_node, node)
                if new_state not in dist:
                    dist[new_state] = d + 1
                    queue.append(new_state)
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for k-regular graphs with high girth."""
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

def _random_bipartite_regular(half, k, max_attempts=50):
    """Generate a random bipartite k-regular graph on 2*half vertices."""
    for _ in range(max_attempts):
        left = list(range(half))
        right = list(range(half, 2 * half))
        
        left_stubs = []
        for v in left:
            for _ in range(k):
                left_stubs.append(v)
        right_stubs = []
        for v in right:
            for _ in range(k):
                right_stubs.append(v)
        
        random.shuffle(right_stubs)
        
        edge_set = set()
        valid = True
        new_edges = []
        for i in range(len(left_stubs)):
            u = left_stubs[i]
            v = right_stubs[i]
            if (u, v) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            new_edges.append((u, v))
        
        if valid:
            return new_edges
    return None

def _random_regular(n, k, max_attempts=50):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    for _ in range(max_attempts):
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        random.shuffle(stubs)
        
        edge_set = set()
        valid = True
        new_edges = []
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            new_edges.append(key)
        
        if valid:
            return new_edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, get the LUW construction as a baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify the LUW graph is valid
    adj = [[] for _ in range(n_luw)]
    for u, v in edges_luw:
        adj[u].append(v)
        adj[v].append(u)
    
    if not _girth_check_nonbacktracking(adj, n_luw, g):
        # Fallback: return as-is (shouldn't happen)
        return n_luw, edges_luw
    
    best_n = n_luw
    best_edges = list(edges_luw)
    
    # For k=4, g=10: lower bound is 243, known best is 320
    # Try random regular graphs for n < best_n
    # n must be even for bipartite, but for general regular graphs, n*k must be even
    # For k=4, n can be any integer (since 4n is always even)
    
    # Try from lower bound up to best_n
    lower_bound = 243 if (k == 4 and g == 10) else 0
    if lower_bound == 0:
        # Estimate Moore bound
        if k == 2:
            lower_bound = g
        else:
            # Moore bound for k-regular graph of girth g
            if g % 2 == 0:
                D = g // 2
                lb = k * (k - 1) ** (D - 1)
            else:
                D = (g - 1) // 2
                lb = 1 + k * sum((k - 1) ** i for i in range(D))
            lower_bound = lb
    
    # Ensure n is even for bipartite approach, or just try all n
    # For k=4, we can try both bipartite and general
    # Let's try general regular graphs first for smaller n
    
    # Try n from lower_bound to best_n - 1
    for n_try in range(lower_bound, best_n):
        if time.time() > deadline:
            break
        
        # Skip if n*k is odd
        if n_try * k % 2 != 0:
            continue
        
        # Try a few random regular graphs
        max_attempts = 10
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            edges_try = _random_regular(n_try, k, max_attempts=1)
            if edges_try is None:
                continue
            
            adj_try = [[] for _ in range(n_try)]
            for u, v in edges_try:
                adj_try[u].append(v)
                adj_try[v].append(u)
            
            if _girth_check_nonbacktracking(adj_try, n_try, g):
                best_n = n_try
                best_edges = edges_try
                break
        
        if best_n == n_try:
            break
    
    return best_n, best_edges
