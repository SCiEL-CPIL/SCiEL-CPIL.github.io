import time
import random
from collections import deque

def _generate_random_k_regular(k, n):
    # Generates a random k-regular graph on n vertices
    # n must be such that k*n is even
    if (k * n) % 2 != 0:
        return None
        
    # Configuration model
    half_edges = k * n // 2
    # Each vertex appears k times in the half-edge list
    # But we need to pair them up.
    # Standard algorithm:
    # 1. Create a list of vertices where each vertex v appears k times.
    # 2. Shuffle the list.
    # 3. Pair up adjacent elements.
    # This creates a multigraph. We need a simple graph.
    # We can use a rejection sampling or a specific algorithm for simple k-regular graphs.
    # For k=5, n=1000, rejection might be slow if many loops/multi-edges.
    # Better: Use the "swap" algorithm or a specialized generator.
    
    # Let's use a simple approach:
    # Start with a random pairing.
    # Then fix loops and multi-edges via swaps.
    
    vertices = []
    for i in range(n):
        vertices.extend([i] * k)
        
    random.shuffle(vertices)
    
    # Pair them
    edges = set()
    for i in range(0, len(vertices), 2):
        u = vertices[i]
        v = vertices[i+1]
        if u == v:
            # Loop, try to fix later or restart
            return None
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            # Multi-edge, try to fix later or restart
            return None
        edges.add((u, v))
        
    # This simple rejection is too slow for n=1000.
    # We need a better generator.
    
    # Alternative: Use the "random permutation" method for 1-regular, then combine?
    # Or use the algorithm from "Random Regular Graphs" literature.
    
    # For simplicity and speed, let's use a different strategy:
    # Start with a cycle 0-1-...-n-1-0. This is 2-regular.
    # Add 3 more perfect matchings randomly.
    
    adj = [set() for _ in range(n)]
    
    # 1. Cycle
    for i in range(n):
        adj[i].add((i+1)%n)
        adj[(i+1)%n].add(i)
        
    # 2. Random matching 1
    # Permute vertices, pair (0,1), (2,3), etc.
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(0, n, 2):
        u = perm[i]
        v = perm[i+1]
        if v in adj[u]:
            # Conflict, retry or handle
            # For now, let's assume conflicts are rare or handle by swapping
            pass
        adj[u].add(v)
        adj[v].add(u)
        
    # 3. Random matching 2
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(0, n, 2):
        u = perm[i]
        v = perm[i+1]
        adj[u].add(v)
        adj[v].add(u)
        
    # 4. Random matching 3
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(0, n, 2):
        u = perm[i]
        v = perm[i+1]
        adj[u].add(v)
        adj[v].add(u)
        
    # Check for simple graph (no multi-edges)
    for u in range(n):
        if len(adj[u]) != k:
            return None
        for v in adj[u]:
            if adj[u].count(v) > 1: # Sets don't have duplicates, so this check is redundant if using sets
                pass
            # But we might have added the same edge twice from different matchings?
            # No, if we use sets, duplicates are ignored.
            # But we need degree exactly k.
            # If an edge was already present, adding it again doesn't change the set.
            # So degree might be < k.
            
    # This approach is flawed because it doesn't ensure degree k if edges collide.
    
    # Let's use a proper random k-regular graph generator.
    # Algorithm:
    # 1. Create list of half-edges.
    # 2. Shuffle.
    # 3. Pair.
    # 4. While there are loops or multi-edges, find a pair of edges that can be swapped to fix them.
    
    return None

def _compute_girth_fast(adjacency, n, early_stop=None):
    min_girth = float('inf')
    if early_stop is None:
        early_stop = 100
    
    for start in range(n):
        if min_girth <= early_stop:
            break
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= early_stop:
                            return min_girth
        if min_girth <= early_stop:
            break
    return min_girth if min_girth != float('inf') else -1

def _find_short_cycle(adjacency, n, max_len):
    # Returns a cycle of length <= max_len, or None
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] > max_len:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len <= max_len:
                        # Reconstruct cycle
                        path1 = []
                        cur = u
                        while cur != start:
                            path1.append(cur)
                            cur = parent[cur]
                        path1.append(start)
                        
                        path2 = []
                        cur = v
                        while cur != start:
                            path2.append(cur)
                            cur = parent[cur]
                        path2.append(start)
                        
                        # Combine
                        # path1 is start -> ... -> u
                        # path2 is start -> ... -> v
                        # Cycle is u -> ... -> start -> ... -> v -> u
                        # We need the actual sequence of vertices.
                        
                        cycle = []
                        cur = u
                        while cur != start:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.append(start)
                        
                        cur = v
                        while cur != start:
                            cycle.append(cur)
                            cur = parent[cur]
                        
                        return cycle
    return None

def _is_2_switch_valid(adj, u, v, u2, v2):
    # Check if switching (u,v) and (u2,v2) to (u,v2) and (u2,v) is valid
    # 1. No loops: u != v2, u2 != v
    if u == v2 or u2 == v:
        return False
    # 2. No multi-edges: (u,v2) not in adj[u], (u2,v) not in adj[u2]
    if v2 in adj[u]:
        return False
    if v in adj[u2]:
        return False
    # 3. The edges being removed must exist
    if v not in adj[u] or v2 not in adj[u2]:
        return False
    return True

def _perform_2_switch(adj, u, v, u2, v2):
    adj[u].remove(v)
    adj[u2].remove(v2)
    adj[v].remove(u)
    adj[v2].remove(u2)
    
    adj[u].add(v2)
    adj[u2].add(v)
    adj[v2].add(u)
    adj[v].add(u2)

def _greedy_switch_improve(adj, n, target_girth, time_limit):
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Find shortest cycle
        cycle = _find_short_cycle(adj, n, target_girth - 1)
        if cycle is None:
            return adj, True
        
        # Pick a random edge from the cycle to switch
        # The cycle is a list of vertices [v0, v1, ..., vk] where vk is connected to v0
        # Edges are (v0,v1), (v1,v2), ..., (vk-1, vk), (vk, v0)
        # Actually, my _find_short_cycle returns a list of vertices in the cycle.
        # Let's say the cycle is C = [c0, c1, ..., c_{L-1}]
        # Edges are (c_i, c_{(i+1)%L})
        
        L = len(cycle)
        if L < 3:
            return adj, False
            
        # Pick a random edge on the cycle
        i = random.randint(0, L-1)
        u = cycle[i]
        v = cycle[(i+1)%L]
        
        # Pick a random edge elsewhere
        # We need to find an edge (u2, v2) such that the switch is valid.
        # To be efficient, we can pick a random vertex u2, then a random neighbor v2.
        # If invalid, try another.
        
        found_switch = False
        for _ in range(100): # Limit attempts per cycle edge
            u2 = random.randint(0, n-1)
            if u2 == u or u2 == v:
                continue
            neighbors = list(adj[u2])
            if not neighbors:
                continue
            v2 = random.choice(neighbors)
            if v2 == u or v2 == v:
                continue
            
            if _is_2_switch_valid(adj, u, v, u2, v2):
                _perform_2_switch(adj, u, v, u2, v2)
                found_switch = True
                break
        
        if not found_switch:
            # Try another edge on the cycle? Or just break?
            # If we can't find a valid switch, the graph might be stuck.
            # Let's try a different edge on the cycle.
            pass
            
    return adj, False

def construct(k, g):
    if k != 5 or g != 9:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
        
    # Target n
    # Lower bound is 428. Best known is 1116.
    # Let's try to find a graph with n around 800-1000.
    
    best_n = float('inf')
    best_edges = None
    
    # Try different n
    for n in [800, 900, 1000, 1100]:
        if (n * k) % 2 != 0:
            continue
            
        # Generate random graph
        # We need a good generator.
        # Let's use the "swap" method to fix a random configuration.
        
        # 1. Create half-edges
        half_edges = []
        for i in range(n):
            half_edges.extend([i] * k)
            
        # 2. Shuffle
        random.shuffle(half_edges)
        
        # 3. Pair
        adj = [set() for _ in range(n)]
        for i in range(0, n*k, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
            # If u==v, it's a loop. We ignore it for now, but degree will be < k.
            # We need to fix this.
            
        # This simple method often results in degrees < k due to loops/multi-edges.
        # We need a robust generator.
        
        # Let's use a different approach:
        # Start with a valid k-regular graph (e.g., cycle + matchings) and randomize it via switches.
        
        # 1. Cycle
        adj = [set() for _ in range(n)]
        for i in range(n):
            adj[i].add((i+1)%n)
            adj[(i+1)%n].add(i)
            
        # 2. Add k-2 random matchings
        for m in range(k-2):
            perm = list(range(n))
            random.shuffle(perm)
            # We need to ensure no multi-edges.
            # If we just add, we might get multi-edges.
            # We can check and retry.
            valid = True
            for i in range(0, n, 2):
                u = perm[i]
                v = perm[i+1]
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                
            if not valid:
                # Retry this matching
                # For simplicity, let's just restart the whole graph generation if this happens often.
                # Or use a more sophisticated matching algorithm.
                # Given the time, let's assume n is large enough that conflicts are rare?
                # For n=1000, k=5, probability of conflict is low.
                pass
        
        # Check if graph is valid (all degrees k, simple)
        valid = True
        for u in range(n):
            if len(adj[u]) != k:
                valid = False
                break
        if not valid:
            continue
            
        # Now improve girth
        adj_improved, success = _greedy_switch_improve(adj, n, g, 20)
        
        if success:
            # Convert to edge list
            edges = []
            for u in range(n):
                for v in adj_improved[u]:
                    if u < v:
                        edges.append((u, v))
                        
            if n < best_n:
                best_n = n
                best_edges = edges
                
    if best_edges is not None:
        return (best_n, best_edges)
        
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW construction
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
