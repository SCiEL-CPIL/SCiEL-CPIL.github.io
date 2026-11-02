import time
import random
import heapq
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _compute_girth(adjacency, n, max_girth=15):
    """Compute the girth using BFS from each vertex."""
    min_girth = max_girth + 1
    for start in range(n):
        # BFS with parent tracking to detect cycles
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
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= max_girth:
                            return min_girth
    return min_girth

def _random_5_regular(n):
    """Generate a random 5-regular graph on n vertices using configuration model."""
    if n * 5 % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * 5)
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    adjacency = [set() for _ in range(n)]
    i = 0
    while i < len(stubs):
        u, v = stubs[i], stubs[i + 1]
        i += 2
        if u == v:
            # Self-loop, retry
            return None
        if v in adjacency[u]:
            # Multi-edge, retry
            return None
        adjacency[u].add(v)
        adjacency[v].add(u)
    
    # Verify regularity
    for i in range(n):
        if len(adjacency[i]) != 5:
            return None
    
    return adjacency

def _switch_improves_girth(adjacency, n, u1, v1, u2, v2, current_girth):
    """Check if switching edges (u1,v1) and (u2,v2) to (u1,u2) and (v1,v2) 
    maintains or improves girth."""
    # Temporarily apply the switch
    adjacency[u1].remove(v1)
    adjacency[v1].remove(u1)
    adjacency[u2].remove(v2)
    adjacency[v2].remove(u2)
    
    # Check for self-loops or multi-edges
    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
        # Revert
        adjacency[u1].add(v1)
        adjacency[v1].add(u1)
        adjacency[u2].add(v2)
        adjacency[v2].add(u2)
        return False
    
    adjacency[u1].add(u2)
    adjacency[u2].add(u1)
    adjacency[v1].add(v2)
    adjacency[v2].add(v1)
    
    # Compute new girth
    new_girth = _compute_girth(adjacency, n, max_girth=12)
    
    # Revert
    adjacency[u1].remove(u2)
    adjacency[u2].remove(u1)
    adjacency[v1].remove(v2)
    adjacency[v2].remove(v1)
    
    adjacency[u1].add(v1)
    adjacency[v1].add(u1)
    adjacency[u2].add(v2)
    adjacency[v2].add(u2)
    
    # Accept if girth doesn't decrease
    return new_girth >= current_girth and new_girth >= 9

def _edge_switching(k, g, n, time_limit=30.0):
    """Use edge switching to try to achieve girth >= g."""
    start_time = time.time()
    
    # Generate initial random 5-regular graph
    max_attempts = 50
    adjacency = None
    for _ in range(max_attempts):
        if time.time() - start_time > time_limit:
            return None
        adjacency = _random_5_regular(n)
        if adjacency is not None:
            break
    
    if adjacency is None:
        return None
    
    # Compute initial girth
    current_girth = _compute_girth(adjacency, n, max_girth=15)
    
    # If already good enough, we're done
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Perform edge switches
    switch_count = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Pick two random edges
        edge_list = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edge_list.append((i, j))
        
        if len(edge_list) < 2:
            break
        
        idx1 = random.randint(0, len(edge_list) - 1)
        idx2 = random.randint(0, len(edge_list) - 1)
        
        if idx1 == idx2:
            continue
        
        u1, v1 = edge_list[idx1]
        u2, v2 = edge_list[idx2]
        
        # Check if edges share a vertex (can't switch)
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
        
        # Try the switch
        if _switch_improves_girth(adjacency, n, u1, v1, u2, v2, current_girth):
            # Apply the switch
            adjacency[u1].remove(v1)
            adjacency[v1].remove(u1)
            adjacency[u2].remove(v2)
            adjacency[v2].remove(u2)
            
            adjacency[u1].add(u2)
            adjacency[u2].add(u1)
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
            
            switch_count += 1
            no_improve_count = 0
            
            # Update girth
            old_girth = current_girth
            current_girth = _compute_girth(adjacency, n, max_girth=15)
            
            if current_girth >= g:
                # Success!
                edges = []
                for i in range(n):
                    for j in adjacency[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with a new random graph
                adjacency = None
                for _ in range(10):
                    if time.time() - start_time > time_limit:
                        return None
                    adjacency = _random_5_regular(n)
                    if adjacency is not None:
                        break
                if adjacency is None:
                    return None
                current_girth = _compute_girth(adjacency, n, max_girth=15)
                if current_girth >= g:
                    edges = []
                    for i in range(n):
                        for j in adjacency[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                no_improve_count = 0
                max_no_improve = 200  # Be more patient after restart
    
    return None

def _greedy_construct(k, g, n, time_limit=10.0):
    # Try to construct a k-regular graph of girth >= g on n vertices
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    
    # Randomize the order of vertices to break symmetry
    vertices = list(range(n))
    random.shuffle(vertices)
    
    # We need to add k * n / 2 edges
    # We process vertices, adding edges to other vertices
    # To ensure regularity, we must be careful.
    # A simple greedy might get stuck with some vertices having < k and others > k (impossible)
    # or some vertices unable to reach k.
    
    # Better approach: 
    # 1. Generate a random k-regular graph (e.g. using configuration model + 2-switches to make simple)
    # 2. Check girth.
    # 3. If girth < g, try to fix by switching.
    # This is complex.
    
    # Alternative: 
    # Start with empty graph.
    # Maintain a list of "active" vertices (degree < k).
    # While active:
    #   Pick u from active.
    #   Pick v from active, v != u.
    #   If (u,v) is not an edge and adding it doesn't create short cycle:
    #      Add edge.
    #      Update degrees.
    #      If degree[u] == k, remove from active.
    #      If degree[v] == k, remove from active.
    #   Else:
    #      Try next v.
    #      If no v works, restart or fail.
    
    # This greedy might fail to complete the graph (stuck with active vertices that cannot connect).
    # To mitigate, we can allow "retries" within the time limit.
    
    def has_short_cycle(u, v, max_depth):
        # BFS from u to see if v is reached within max_depth
        # If yes, adding (u,v) creates a cycle of length <= max_depth + 1
        # We need cycle >= g, so path length must be >= g - 1
        # So we reject if dist(u,v) < g - 1
        if u == v: return True
        visited = {u: 0}
        queue = [(u, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist >= max_depth:
                continue
            for neighbor in adjacency[node]:
                if neighbor == v:
                    return True # Found v, so path exists
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    active = list(range(n))
    random.shuffle(active)
    
    # We need to be careful with the "active" list management
    # Using a set for active vertices for O(1) removal?
    # But we need to pick randomly. List is fine for n~1000.
    
    while active:
        if time.time() - start_time > time_limit:
            return None
            
        # Pick a vertex u from active
        u_idx = random.randint(0, len(active) - 1)
        u = active[u_idx]
        
        # Find a valid v
        found = False
        # Shuffle active to avoid bias
        active_copy = active[:]
        random.shuffle(active_copy)
        
        for v in active_copy:
            if v == u:
                continue
            if v in adjacency[u]:
                continue
            
            # Check if adding (u,v) creates a short cycle
            # We need to ensure no path of length < g-1 between u and v
            if has_short_cycle(u, v, g - 2):
                continue
            
            # Add edge
            adjacency[u].add(v)
            adjacency[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
            # Update active list
            # Remove u if degree == k
            # Remove v if degree == k
            # This is inefficient if we do it every time.
            # Instead, we can just check degrees when picking.
            # But `active` contains vertices with degree < k.
            # If we add an edge, their degrees increase.
            # If they reach k, they should be removed.
            
            found = True
            break
            
        if not found:
            # Failed to find a partner for u
            # Restart
            return None
            
        # Clean up active list
        # This is O(n). Acceptable for n=1000.
        new_active = []
        for node in active:
            if degrees[node] < k:
                new_active.append(node)
        active = new_active
        
    # Verify final graph
    # Check degrees
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
            
    # Check girth (optional, but good for safety)
    # The greedy check should have ensured it.
    
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges

def construct(k, g):
    # For the specific case k=5, g=9, try edge switching
    if k == 5 and g == 9:
        # Try a few sizes
        # The best known is 1116.
        # We try slightly larger to increase probability of success.
        target_ns = [1116, 1150, 1200, 1300, 1500]
        
        # Time budget: 80s for edge switching, 10s for fallback
        time_per_attempt = 15.0
        
        for n in target_ns:
            res = _edge_switching(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
        # Try greedy as well
        for n in target_ns:
            res = _greedy_construct(k, g, n, time_limit=5.0)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
