import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    # Fallback: seed program logic (L-U-W construction)
    def seed_construct(k, g):
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

    # Girth check using BFS from each vertex
    def check_girth(adj, n, g):
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Generate a random k-regular graph using configuration model with repair
    def generate_random_regular(n, k, rng):
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Randomly pair stubs
        rng.shuffle(stubs)
        edges = []
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            edges.append((u, v))
        
        # Build adjacency and check for multi-edges/self-loops
        # Repair by swapping
        max_attempts = 100
        for attempt in range(max_attempts):
            # Build edge count matrix to detect multi-edges
            edge_count = {}
            self_loops = 0
            for u, v in edges:
                if u == v:
                    self_loops += 1
                else:
                    key = (min(u, v), max(u, v))
                    edge_count[key] = edge_count.get(key, 0) + 1
            
            if self_loops == 0 and all(c == 1 for c in edge_count.values()):
                return edges
            
            # Find a problematic edge (self-loop or multi-edge)
            problem_idx = -1
            if self_loops > 0:
                for i, (u, v) in enumerate(edges):
                    if u == v:
                        problem_idx = i
                        break
            else:
                for i, (u, v) in enumerate(edges):
                    key = (min(u, v), max(u, v))
                    if edge_count.get(key, 0) > 1:
                        problem_idx = i
                        break
            
            if problem_idx == -1:
                return edges
            
            u1, v1 = edges[problem_idx]
            
            # Find another edge to swap with
            swapped = False
            for j in range(n * k // 2):
                if j == problem_idx:
                    continue
                u2, v2 = edges[j]
                if u2 == v2:
                    continue
                
                # Try swapping: (u1, v2) and (u2, v1) or (u1, u2) and (v1, v2)
                # We need to avoid creating new self-loops or multi-edges
                for new_edges in [((u1, v2), (u2, v1)), ((u1, u2), (v1, v2))]:
                    (a, b), (c, d) = new_edges
                    if a == b or c == d:
                        continue
                    # Check if this creates multi-edges
                    new_count = dict(edge_count)
                    old_key1 = (min(u1, v1), max(u1, v1)) if u1 != v1 else None
                    old_key2 = (min(u2, v2), max(u2, v2))
                    if old_key1:
                        new_count[old_key1] -= 1
                        if new_count[old_key1] == 0:
                            del new_count[old_key1]
                    if old_key2:
                        new_count[old_key2] -= 1
                        if new_count[old_key2] == 0:
                            del new_count[old_key2]
                    
                    new_key1 = (min(a, b), max(a, b))
                    new_key2 = (min(c, d), max(c, d))
                    if new_key1 in new_count and new_count[new_key1] >= 1:
                        continue
                    if new_key2 in new_count and new_count[new_key2] >= 1:
                        continue
                    if new_key1 == new_key2:
                        continue
                    
                    # Apply swap
                    edges[problem_idx] = (a, b)
                    edges[j] = (c, d)
                    swapped = True
                    break
                if swapped:
                    break
            
            if not swapped:
                # Try again with a fresh shuffle
                rng.shuffle(stubs)
                edges = []
                for i in range(0, n * k, 2):
                    u, v = stubs[i], stubs[i + 1]
                    edges.append((u, v))
        
        # Final check
        edge_count = {}
        self_loops = 0
        for u, v in edges:
            if u == v:
                self_loops += 1
            else:
                key = (min(u, v), max(u, v))
                edge_count[key] = edge_count.get(key, 0) + 1
        
        if self_loops == 0 and all(c == 1 for c in edge_count.values()):
            return edges
        
        return None

    # Local search: try to improve girth by edge swaps
    def local_search(adj, n, k, g, edges, rng, time_limit):
        start = time.time()
        best_girth = float('inf')
        
        # First, compute current girth
        current_girth = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < current_girth:
                            current_girth = cycle_len
        
        if current_girth >= g:
            return edges, current_girth
        
        # Try random edge swaps to improve girth
        while time.time() - start < time_limit:
            # Pick two random edges
            i = rng.randint(0, len(edges) - 1)
            j = rng.randint(0, len(edges) - 1)
            if i == j:
                continue
            
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            
            # Try swap
            for new_edges in [((u1, v2), (u2, v1)), ((u1, u2), (v1, v2))]:
                (a, b), (c, d) = new_edges
                if a == b or c == d:
                    continue
                
                # Check for multi-edges
                if (a, c) in [(x, y) for x, y in edges if x != y] or (a, d) in [(x, y) for x, y in edges if x != y]:
                    continue
                if (c, a) in [(x, y) for x, y in edges if x != y] or (d, a) in [(x, y) for x, y in edges if x != y]:
                    continue
                
                # Apply swap temporarily
                old_e1, old_e2 = edges[i], edges[j]
                edges[i] = (a, b)
                edges[j] = (c, d)
                
                # Rebuild adjacency
                new_adj = [[] for _ in range(n)]
                for u, v in edges:
                    new_adj[u].append(v)
                    new_adj[v].append(u)
                
                # Check girth
                new_girth = float('inf')
                for s in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[s] = 0
                    queue = deque([s])
                    while queue:
                        u = queue.popleft()
                        if dist[u] >= g // 2:
                            continue
                        for v in new_adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < new_girth:
                                    new_girth = cycle_len
                
                if new_girth >= g:
                    return edges, new_girth
                elif new_girth > current_girth:
                    current_girth = new_girth
                    adj = new_adj
                else:
                    # Revert
                    edges[i] = old_e1
                    edges[j] = old_e2
        
        return edges, current_girth

    # Main search: try increasing n from moore_bound
    rng = random.Random(42)
    best_result = None
    
    # Start from a reasonable n and go up
    # For k=5, g=9, moore_bound = 426
    # Try n = moore_bound, moore_bound+1, ...
    
    # To be efficient, we'll try a few n values with time limits
    n_candidates = []
    for n in range(moore_bound, min(moore_bound + 200, 1000)):
        n_candidates.append(n)
    
    for n in n_candidates:
        if time.time() - start_time > deadline - 5:
            break
        
        # Try to generate and improve a random k-regular graph
        edges = generate_random_regular(n, k, rng)
        if edges is None:
            continue
        
        # Build adjacency
        adj = build_adj(edges, n)
        
        # Check if girth is already sufficient
        if check_girth(adj, n, g):
            best_result = (n, edges)
            break
        
        # Try local search
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        time_per_n = max(1.0, remaining_time / (len(n_candidates) - n_candidates.index(n) - 1))
        edges, girth = local_search(adj, n, k, g, edges, rng, time_per_n)
        
        if girth >= g:
            best_result = (n, edges)
            break
    
    # If no result found, fall back to seed
    if best_result is None:
        best_result = seed_construct(k, g)
    
    return best_result
