import time
from itertools import combinations

# EVOLVE-BLOCK-START
import sys

def has_c6(adj, n):
    """Check if graph has a 6-cycle using BFS-based approach."""
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                continue
            # Look for path of length 4 from v to some neighbor of u (not v)
            # v - a - b - c - d where d in N(u) \ {v}
            # and a,b,c,d all distinct and not u
            visited = {u, v}
            # BFS from v, tracking paths of length up to 4
            # We want to find d in N(u)\{v} reachable from v in exactly 4 steps
            # with all intermediate vertices distinct from u,v and each other
            queue = [(v, 0, frozenset())]
            while queue:
                curr, dist, path = queue.pop(0)
                if dist == 4:
                    if curr in adj[u] and curr != v:
                        return True
                    continue
                if dist < 4:
                    for nxt in adj[curr]:
                        if nxt in visited or nxt in path:
                            continue
                        queue.append((nxt, dist + 1, path | {nxt}))
    return False

def is_c6_free(adj, n):
    return not has_c6(adj, n)

def check_c6_fast(adj, u, v, n):
    """Fast check: does adding edge (u,v) create a C6?
    A C6 through (u,v): u-v-a-b-c-d-u
    Need d in N(u)\{v}, and path v-a-b-c-d of length 4 with distinct internal nodes.
    Equivalently: exists d in N(u)\{v} such that dist_{G - {u,v}}(v, d) == 4
    with a simple path of length 4.
    """
    # For each d in N(u) \ {v}, check if there's a path of length exactly 4 from v to d
    # avoiding u and v (and with distinct internal vertices)
    for d in adj[u]:
        if d == v:
            continue
        # BFS from v to d with exactly 4 edges, avoiding u,v
        # Use DP: dp[step][node] = set of nodes reachable in 'step' steps from v
        # with distinct vertices (excluding u,v)
        # Simpler: enumerate paths of length 4 from v
        # v -> a -> b -> c -> d
        for a in adj[v]:
            if a == u or a == v or a == d:
                continue
            for b in adj[a]:
                if b == u or b == v or b == d or b == a:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == d or c == a or c == b:
                        continue
                    if d in adj[c]:
                        return True
    return False

def greedy_build(n=40, seed=42, time_limit=145.0):
    """Improved greedy construction + local search."""
    import random
    random.seed(seed)
    
    start = time.time()
    deadline = start + time_limit - 5.0  # safety margin
    
    # Start with a better seed: try to build from scratch
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    def edge_count():
        return sum(len(a) for a in adj) // 2
    
    # Phase 1: Systematically try to add all possible edges in random order
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 10:  # leave time for local search
            break
        if v in adj[u]:
            continue
        if not check_c6_fast(adj, u, v, n):
            add_edge(u, v)
    
    best = edge_count()
    
    # Phase 2: Local search with edge swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try random edge swaps: remove one edge, try to add another
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random existing edge to remove
            existing_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        existing_edges.append((u, v))
            if not existing_edges:
                break
            ru, rv = random.choice(existing_edges)
            
            # Try to add a random non-edge
            attempts = 0
            while attempts < 20:
                a, b = random.randint(0, n - 1), random.randint(0, n - 1)
                if a == b or b in adj[a]:
                    attempts += 1
                    continue
                if (a, b) == (ru, rv) or (b, a) == (ru, rv):
                    attempts += 1
                    continue
                
                # Temporarily remove (ru, rv) and check if (a, b) can be added
                remove_edge(ru, rv)
                
                if not check_c6_fast(adj, a, b, n):
                    add_edge(a, b)
                    new_count = edge_count()
                    if new_count > best:
                        best = new_count
                        improved = True
                    else:
                        remove_edge(a, b)
                else:
                    # Restore (ru, rv)
                    add_edge(ru, rv)
                    break
                
                # If we kept the swap, don't restore
                break
            else:
                # If we exhausted attempts, restore (ru, rv)
                add_edge(ru, rv)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges, best

def construct():
    start = time.time()
    deadline = start + 145.0  # 150s limit, 5s safety
    
    try:
        edges, count = greedy_build(n=40, seed=42, time_limit=140.0)
    except Exception:
        edges, count = _fallback(), 80
    
    if time.time() > deadline - 2.0 or count < 80:
        return _fallback()
    
    # Validate
    adj = [set() for _ in range(40)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    if not is_c6_free(adj, 40):
        return _fallback()
    
    return edges

def _fallback():
    """8 disjoint copies of K5."""
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    return edges
# EVOLVE-BLOCK-END
