"""New approach: Simulated Annealing with local girth checks."""
import random
import math
from collections import deque

def _random_regular_graph(n, k, rng):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    # Build stub list
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    rng.shuffle(stubs)
    # Pair up stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # self-loop, retry
        if (u, v) in edges or (v, u) in edges:
            return None  # multi-edge, retry
        edges.add((min(u, v), max(u, v)))
    return list(edges)

def _build_adj(n, edges):
    """Build adjacency list from edges."""
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_short_cycle(adj, girth):
    """Check if graph has any cycle shorter than girth."""
    n = len(adj)
    for start in range(n):
        # BFS to detect cycles shorter than girth
        # Use a more efficient method: check all cycles up to girth-1
        # For each vertex, do limited DFS
        def dfs(node, depth, path):
            if depth >= girth - 1:
                return False
            for nb in adj[node]:
                if nb == path[0] and depth >= 2:
                    return True
                if nb not in path:
                    if dfs(nb, depth + 1, path + [nb]):
                        return True
            return False
        if dfs(start, 0, [start]):
            return True
    return False

def _check_swap_safe(adj, e1, e2, limit):
    """
    Check if swapping e1 and e2 creates any cycle of length < limit.
    e1 = (u1, v1), e2 = (u2, v2).
    New edges: (u1, u2) and (v1, v2) or (u1, v2) and (v1, u2).
    We check both options.
    Returns True if a safe swap exists, and which one.
    """
    u1, v1 = e1
    u2, v2 = e2
    
    # Option 1: (u1, u2) and (v1, v2)
    # Option 2: (u1, v2) and (v1, u2)
    
    def is_safe_swap(new_e1, new_e2):
        a, b = new_e1
        c, d = new_e2
        if a == b or c == d:
            return False
        # Check if new edges already exist in adj
        if b in adj[a] or d in adj[c]:
            return False
        
        # Temporarily remove old edges and add new ones for checking?
        # No, we can just check paths in the graph with old edges removed.
        # A new cycle must use at least one new edge.
        # So we check if there is a path from a to b in G \ {e1, e2} of length < limit - 1.
        # And path from c to d in G \ {e1, e2} of length < limit - 1.
        
        # To be efficient, we can just do BFS from a to b and c to d.
        # We must exclude e1 and e2 from the BFS graph.
        
        # Create a view of adj without e1 and e2
        # For speed, we just check neighbors manually in BFS.
        
        def bfs_shortest_path(src, dst, excluded_edges):
            if src == dst:
                return 0
            visited = {src: 0}
            queue = deque([(src, 0)])
            while queue:
                node, dist = queue.popleft()
                if dist >= limit - 1:
                    break
                for nb in adj[node]:
                    # Check if edge (node, nb) is excluded
                    edge_key = (min(node, nb), max(node, nb))
                    if edge_key in excluded_edges:
                        continue
                    if nb not in visited:
                        if nb == dst:
                            return dist + 1
                        visited[nb] = dist + 1
                        queue.append((nb, dist + 1))
            return float('inf')

        excluded = {e1, e2}
        
        # Check cycle through new_e1
        path_len_1 = bfs_shortest_path(a, b, excluded)
        if path_len_1 < limit - 1:
            return False
        
        # Check cycle through new_e2
        path_len_2 = bfs_shortest_path(c, d, excluded)
        if path_len_2 < limit - 1:
            return False
            
        return True

    # Try Option 1
    if is_safe_swap((u1, u2), (v1, v2)):
        return ((u1, u2), (v1, v2))
    
    # Try Option 2
    if is_safe_swap((u1, v2), (v1, u2)):
        return ((u1, v2), (v1, u2))
        
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    if k != 3:
        # Fallback for other k (not optimized)
        return _construct_luw(k, g)
    
    girth_target = g
    rng = random.Random(42)
    
    # Try various sizes starting from Moore bound
    # Moore bound for 3-regular, girth 16 is 382.
    # We expect n to be around 900-1000.
    
    best_n = None
    best_edges = None
    
    # We will try to find a graph for increasing n.
    # Start from a reasonable estimate.
    # Random 3-reg graphs have girth ~ log2(n).
    # For g=16, n ~ 2^16 = 65536 is expected for RANDOM.
    # But we are optimizing.
    # Exoo et al. found 936.
    
    # Let's try n from 400 to 1200 in steps.
    # If we find one, we try to minimize n.
    
    for n in range(400, 1200, 10):
        if n % 2 != 0:
            continue
            
        # Try to find a valid graph of size n
        found = False
        for trial in range(5): # 5 restarts per n
            edges = _random_regular_graph(n, k, rng)
            if edges is None:
                continue
            
            adj = _build_adj(n, edges)
            
            # Initial girth check
            if _has_short_cycle(adj, girth_target):
                # Run Simulated Annealing
                adj = _sa_optimize(adj, girth_target, n, rng)
                
            if adj is not None and not _has_short_cycle(adj, girth_target):
                # Success
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = [(min(u, v), max(u, v)) for u, v in adj.items() for v in adj[u] if u < v]
                    found = True
                    break # Found for this n, try smaller n? No, we are iterating n upwards.
                    # Wait, we want MIN n. So we should iterate n from small to large.
                    # If we find one at n=400, we should stop?
                    # But 400 might be impossible.
                    # The loop goes 400, 410, ...
                    # So if we find one at 930, we record it.
                    # If we find one at 920, we record it.
                    # We should break out of n loop if we hit the Moore bound?
                    # No, just keep going to find the smallest valid n.
        
        # If we found a valid graph at this n, we can't stop immediately 
        # because a smaller n (already passed) might have failed but maybe 
        # we should have tried harder?
        # Actually, if we iterate n from small to large, the first valid n we find
        # is the smallest we've found so far.
        # But we might have skipped a valid n=400 because we didn't try hard enough.
        # So we should try hard for small n.
        
        if found:
            # We found a graph of size n.
            # Can we find a smaller one?
            # We've already tried smaller n's.
            # So this is the best we have.
            # We can stop if we are confident?
            # Or continue to see if we can find an even smaller one?
            # No, we are iterating UP. So smaller ones are already tried.
            # So we can break?
            # Wait, if we tried n=400 and failed, and n=410 failed...
            # and n=930 succeeded.
            # The answer is 930.
            # We can break.
            break

    if best_n is not None:
        return best_n, best_edges
    
    # Fallback
    return _construct_luw(k, g)

def _sa_optimize(adj, girth_target, n, rng, max_iterations=5000):
    """
    Simulated Annealing to increase girth.
    """
    current_girth = _estimate_girth(adj, girth_target)
    
    if current_girth >= girth_target:
        return adj
    
    T = 10.0
    cooling = 0.99
    
    for i in range(max_iterations):
        # Pick random edge
        all_edges = [(u, v) for u, v in adj.items() for v in adj[u] if u < v]
        if not all_edges:
            break
        e1 = rng.choice(all_edges)
        
        # Pick random non-adjacent edge
        # For simplicity, pick random e2 and check adjacency
        e2 = rng.choice(all_edges)
        if e2 == e1:
            continue
        if len(set(e1) & set(e2)) > 0:
            continue # Adjacent, skip
        
        new_edges = _check_swap_safe(adj, e1, e2, girth_target)
        if new_edges is None:
            continue
        
        ne1, ne2 = new_edges
        
        # Perform swap
        adj_new = {v: set(adj[v]) for v in adj}
        u1, v1 = e1
        u2, v2 = e2
        
        adj_new[u1].discard(v1)
        adj_new[v1].discard(u1)
        adj_new[u2].discard(v2)
        adj_new[v2].discard(u2)
        
        a, b = ne1
        c, d = ne2
        adj_new[a].add(b)
        adj_new[b].add(a)
        adj_new[c].add(d)
        adj_new[d].add(c)
        
        # Calculate new girth estimate
        new_girth = _estimate_girth(adj_new, girth_target)
        
        # Accept if better, or with prob exp(-delta/T)
        if new_girth > current_girth:
            adj = adj_new
            current_girth = new_girth
            if current_girth >= girth_target:
                return adj
        else:
            delta = current_girth - new_girth
            if delta > 0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    adj = adj_new
                    current_girth = new_girth
        
        T *= cooling
        if T < 0.01:
            break
            
    return adj

def _estimate_girth(adj, limit):
    """
    Estimate girth by finding the shortest cycle.
    If no cycle < limit found, return limit.
    """
    n = len(adj)
    for start in range(n):
        # BFS to find shortest cycle from start
        # This is O(N * 2^g) roughly
        # We only need to know if it's < limit
        def dfs(node, depth, path):
            if depth >= limit - 1:
                return limit
            min_g = limit
            for nb in adj[node]:
                if nb == path[0] and depth >= 2:
                    min_g = min(min_g, depth + 1)
                if nb not in path:
                    res = dfs(nb, depth + 1, path + [nb])
                    min_g = min(min_g, res)
            return min_g
        
        g = dfs(start, 0, [start])
        if g < limit:
            return g
            
    return limit

def _construct_luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
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
