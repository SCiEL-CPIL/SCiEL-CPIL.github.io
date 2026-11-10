"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses random configuration model + local search (2-switches) for k=3.
    Falls back to Lazebnik-Ustimenko-Woldar construction if local search fails.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Moore bound calculation for lower bound estimation
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    if k == 3:
        # For g=16, half_g = 8.
        # n >= 2 * (1 + 2 + 4 + ... + 2^7) = 2 * (2^8 - 1) = 510.
        # We aim for n slightly above this, e.g., 600-1000.
        moore_bound = 2 * ((2 ** (g // 2)) - 1)
    else:
        # General approximation
        moore_bound = 2 * sum((k-1)**i for i in range(g // 2))
        
    # Start n at a value slightly above Moore bound to allow for "waste" in random graph
    # But not too high to keep search space manageable.
    # For g=16, n=1000 is a good target.
    target_n = max(moore_bound + 100, 400)
    
    # Ensure n is even for 3-regular graph
    if target_n % 2 != 0:
        target_n += 1
        
    best_n = float('inf')
    best_edges = None
    
    # Try increasing n if initial attempts fail
    for n_attempt in range(target_n, target_n + 400, 50):
        if time.time() > deadline:
            break
        if n_attempt % 2 != 0:
            continue
            
        # Try to find a valid graph for this n
        success, edges = _try_random_search(n_attempt, k, g, deadline)
        if success:
            return n_attempt, edges
            
    # Fallback: Lazebnik-Ustimenko-Woldar construction
    return _lnw_construct(k, g)

def _try_random_search(n, k, g, deadline):
    """
    Attempt to construct a k-regular graph on n vertices with girth >= g
    using random generation and local search.
    """
    if k != 3:
        # This specific heuristic is tuned for k=3.
        # For other k, we might need different strategies, but for this task k=3.
        pass
        
    # Generate a random 3-regular graph
    adj = _generate_random_3reg(n)
    if adj is None:
        return False, None
        
    # Initial check
    current_girth = _calculate_girth(adj, n)
    if current_girth >= g:
        return True, _adj_to_edges(adj)
        
    # Local search: 2-switches
    # A 2-switch replaces edges (u,v) and (x,y) with (u,x) and (v,y) or (u,y) and (v,x)
    # such that no loops or multi-edges are created.
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline:
            return False, None
            
        # Pick a random edge (u, v)
        u = random.randint(0, n - 1)
        neighbors_u = list(adj[u])
        if not neighbors_u:
            continue
        v = random.choice(neighbors_u)
        
        # Pick another random edge (x, y) not sharing vertices with (u,v)
        # To increase chance of breaking cycles, we could pick edges from short cycles,
        # but random is simpler and often effective.
        x = random.randint(0, n - 1)
        if x in (u, v):
            continue
        neighbors_x = list(adj[x])
        if not neighbors_x:
            continue
        y = random.choice(neighbors_x)
        if y in (u, v) or y == x:
            continue
            
        # Ensure (u,v) and (x,y) are distinct edges
        if (u, v) == (x, y) or (u, v) == (y, x):
            continue
            
        # Try swap: remove (u,v), (x,y); add (u,x), (v,y)
        # Check for multi-edges
        if x in adj[u] or y in adj[v]:
            continue
        # Check for loops (already handled by x!=u, y!=v etc, but explicit is good)
        if u == x or v == y:
            continue
            
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].add(x)
        adj[x].add(u)
        adj[v].add(y)
        adj[y].add(v)
        
        # Check girth
        # Optimization: Only check if swap might have improved girth?
        # Hard to know without checking.
        # Full girth check is O(n * girth). For n~1000, girth~16, this is fast.
        current_girth = _calculate_girth(adj, n)
        if current_girth >= g:
            return True, _adj_to_edges(adj)
            
    return False, None

def _generate_random_3reg(n):
    """
    Generate a random 3-regular graph on n vertices using configuration model.
    Returns adjacency list (list of sets) or None if failed (shouldn't happen for large n).
    """
    if n < 4 or n % 2 != 0:
        return None
        
    stubs = []
    for i in range(n):
        stubs.extend([i] * 3)
        
    # Shuffle and pair
    random.shuffle(stubs)
    adj = [set() for _ in range(n)]
    valid = True
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Self-loop, try to fix by swapping with another pair? 
            # For simplicity, restart if too many failures, but rare for large n.
            # We'll just mark invalid and retry generation.
            valid = False
            break
        if v in adj[u]:
            # Multi-edge
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        
    if not valid:
        # Retry a few times
        for _ in range(10):
            stubs = []
            for i in range(n):
                stubs.extend([i] * 3)
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                break
        if not valid:
            return None
            
    # Verify degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    return adj

def _calculate_girth(adj, n):
    """
    Calculate the girth (length of shortest cycle) of a graph.
    Uses BFS from each vertex to find shortest cycle.
    O(n * m) roughly, but with early termination.
    """
    min_cycle = n + 1
    
    for start in range(n):
        # BFS from start
        # dist[v] = shortest distance from start to v
        # parent[v] = parent in BFS tree
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if dist[u] >= min_cycle:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # If v is already visited and v is not parent of u
                    # and u is not parent of v (handled by non-backtracking logic implicitly in BFS tree check?)
                    # Actually, standard BFS finds shortest path.
                    # A cycle is formed if we encounter a visited node v from u,
                    # where v is not the parent of u.
                    # The length of the cycle is dist[u] + dist[v] + 1.
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            
    return min_cycle if min_cycle <= n else n + 1

def _adj_to_edges(adj):
    """Convert adjacency list to list of edges (u, v) with u < v."""
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _lnw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
