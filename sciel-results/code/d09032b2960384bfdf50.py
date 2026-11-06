import time
import sys
import random

DEADLINE_MARGIN = 15.0  # seconds before hard limit to start returning
HARD_LIMIT = 135.0      # seconds

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and (w in adj[u]):
                    return False
    return True

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with bitmasks and vertex ordering."""
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    colors = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        forbidden = 0
        for nb in adj[v]:
            c = colors[nb]
            if c >= 0:
                forbidden |= (1 << c)
        
        for c in range(3):
            if forbidden & (1 << c):
                continue
            colors[v] = c
            if backtrack(idx + 1):
                return True
            colors[v] = -1
        
        return False
    
    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Start with the Mycielski of C5 (Grötzsch graph) on vertices 0-10
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    
    def remove_edge(u, v):
        if u in adj[v]:
            adj[u].discard(v)
            adj[v].discard(u)
    
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    
    # Mycielski layer: 5-9 where 5+i connects to neighbors of i in C5
    for i in range(5):
        neighbors = [(i + 1) % 5, (i - 1) % 5]
        for nb in neighbors:
            add_edge(5 + i, nb)
    
    # Universal vertex 10 connects to all 5-9
    for i in range(5):
        add_edge(10, 5 + i)
    
    # Vertices 11-15 start isolated
    # Current edge count: C5 has 5 edges, Mycielski layer has 10 edges, vertex 10 has 5 edges = 20 edges
    
    edge_count = sum(len(adj[u]) for u in range(n)) // 2
    
    def can_add_edge(u, v):
        if u in adj[v]:
            return False
        # Check triangle-free
        common = adj[u] & adj[v]
        if common:
            return False
        return True
    
    # Greedy fill: add as many edges as possible while maintaining triangle-free
    # Since the base is not 3-colorable, adding edges preserves non-3-colorability
    changed = True
    while changed:
        changed = False
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u + 1, n):
                if can_add_edge(u, v):
                    add_edge(u, v)
                    edge_count += 1
                    changed = True
        if time.time() > deadline:
            break
    
    # Now try edge swap moves: remove one edge, add two edges
    # This can escape local optima
    best_edge_count = edge_count
    best_adj = [set(s) for s in adj]
    
    random.seed(42)
    
    def try_swap():
        nonlocal edge_count, best_edge_count, best_adj
        if time.time() > deadline:
            return False
        
        # Pick a random existing edge to remove
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if not edges:
            return False
        
        u, v = random.choice(edges)
        remove_edge(u, v)
        edge_count -= 1
        
        # Try to add two edges that were previously blocked
        candidates = []
        for a in range(n):
            for b in range(a + 1, n):
                if a not in adj[b] and not (adj[a] & adj[b]):
                    candidates.append((a, b))
        
        random.shuffle(candidates)
        
        added = 0
        added_edges = []
        for a, b in candidates:
            if added >= 2:
                break
            if a not in adj[b] and not (adj[a] & adj[b]):
                add_edge(a, b)
                edge_count += 1
                added += 1
                added_edges.append((a, b))
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = [set(s) for s in adj]
        
        # Revert the swap
        for a, b in added_edges:
            remove_edge(a, b)
        add_edge(u, v)
        edge_count += 1
        
        return added == 2
    
    # Try many swap moves
    while time.time() < deadline:
        try_swap()
    
    # Restore best found
    if best_edge_count > edge_count:
        adj = [set(s) for s in best_adj]
        edge_count = best_edge_count
    
    # Final greedy fill
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if can_add_edge(u, v):
                    add_edge(u, v)
                    edge_count += 1
                    changed = True
    
    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
