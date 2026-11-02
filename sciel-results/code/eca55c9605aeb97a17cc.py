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

def _compute_girth(adjacency, n):
    """Compute the girth of the graph using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        # BFS to find shortest cycle containing start
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
                    # Found a back edge, cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth <= 3:
                            return min_girth
        if min_girth <= 3:
            break
    return min_girth if min_girth != float('inf') else -1

def _find_short_cycle(adjacency, n, max_girth):
    """Find a cycle of length <= max_girth if one exists. Returns list of vertices or None."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a back edge
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len <= max_girth:
                        # Reconstruct cycle
                        cycle = []
                        # Walk from u back to LCA
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                        
                        # Find LCA
                        set_u = set(path_u)
                        lca = None
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                        
                        if lca is None:
                            continue
                            
                        # Build cycle
                        idx_u = path_u.index(lca)
                        idx_v = path_v.index(lca)
                        cycle = path_u[:idx_u+1][::-1] + path_v[:idx_v][::-1]
                        
                        # Remove duplicate lca
                        if cycle[0] == cycle[-1]:
                            cycle = cycle[:-1]
                        
                        if len(cycle) > 1:
                            return cycle
    return None

def _is_2_switch_valid(adjacency, u, v, w, x, g):
    """Check if performing 2-switch (u,v)->(u,w),(v,x) or similar doesn't create short cycles.
    We need to check that removing edges (u,v) and (w,x) and adding (u,w) and (v,x)
    doesn't create any cycle shorter than g."""
    # The 2-switch: remove (u,v) and (w,x), add (u,w) and (v,x)
    # This is valid if u!=v, w!=x, u!=w, v!=x, and the new edges don't create short cycles
    
    # Check that the new edges don't create short cycles
    # We need to verify that in the graph after the switch, there's no cycle < g
    
    # For efficiency, we can check locally:
    # The only way a new short cycle can be created is if it uses one or both of the new edges
    # A cycle using (u,w): path from u to w in G - {(u,v),(w,x)} with length < g-1
    # A cycle using (v,x): path from v to x in G - {(u,v),(w,x)} with length < g-1
    # A cycle using both: path from u to x and w to v (or u to v and w to x) in G - {(u,v),(w,x)}
    
    # For simplicity, let's just do a full BFS check but limit depth
    
    # Temporarily modify adjacency
    # Actually, let's just check if adding (u,w) and (v,x) creates short cycles
    # by checking distances in the graph without (u,v) and (w,x)
    
    # Check if there's a path from u to w of length < g-1 in G - {(u,v),(w,x)}
    # and similarly for v to x
    
    def has_short_path(adj, src, dst, max_dist, excluded_edges):
        """Check if there's a path from src to dst of length <= max_dist avoiding excluded_edges."""
        if src == dst:
            return 0
        visited = {src: 0}
        queue = deque([(src, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= max_dist:
                continue
            for neighbor in adj[node]:
                # Skip excluded edges
                if (node, neighbor) in excluded_edges or (neighbor, node) in excluded_edges:
                    continue
                if neighbor == dst:
                    return dist + 1
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return -1
    
    excluded = {(u, v), (v, u), (w, x), (x, w)}
    
    # Check if u-w path exists with length < g-1
    if has_short_path(adjacency, u, w, g - 2, excluded) != -1:
        return False
    
    # Check if v-x path exists with length < g-1
    if has_short_path(adjacency, v, x, g - 2, excluded) != -1:
        return False
    
    return True

def _generate_simple_k_regular(k, n, max_attempts=100):
    """Generate a simple k-regular graph on n vertices using configuration model with retries."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(max_attempts):
        # Create configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair up stubs to form edges
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if valid:
            # Verify all degrees are exactly k
            for i in range(n):
                if len(adjacency[i]) != k:
                    valid = False
                    break
            if valid:
                return adjacency
    
    return None

def _switching_construct(k, g, n, time_limit=100.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate a random k-regular graph
    adjacency = _generate_simple_k_regular(k, n, max_attempts=50)
    if adjacency is None:
        return None
    
    # Step 2: Use 2-switches to increase girth
    # Current girth
    current_girth = _compute_girth(adjacency, n)
    
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Try to increase girth using 2-switches
    max_iterations = 20000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Find a short cycle
        short_cycle = _find_short_cycle(adjacency, n, g - 1)
        if short_cycle is None:
            # No short cycle found, we're done
            edges = []
            for i in range(n):
                for j in adjacency[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        # Try to break the short cycle using a 2-switch
        cycle_len = len(short_cycle)
        if cycle_len < 4:
            # Triangle, hard to break with 2-switch
            # Try random 2-switch instead
            u = random.randint(0, n - 1)
            v = random.choice(list(adjacency[u]))
            # Find w, x such that (w,x) is an edge and w,x not in {u,v}
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                for x in adjacency[w]:
                    if x == u or x == v or x == w:
                        continue
                    if x in adjacency[u] or w in adjacency[v]:
                        continue
                    candidates.append((w, x))
            if not candidates:
                continue
            w, x = random.choice(candidates)
            
            # Check if switch is valid
            if _is_2_switch_valid(adjacency, u, v, w, x, g):
                # Perform switch
                adjacency[u].remove(v)
                adjacency[v].remove(u)
                adjacency[w].remove(x)
                adjacency[x].remove(w)
                adjacency[u].add(w)
                adjacency[w].add(u)
                adjacency[v].add(x)
                adjacency[x].add(v)
        else:
            # Pick two non-adjacent edges in the cycle
            attempts = 0
            while attempts < 10:
                attempts += 1
                edge1_idx = random.randint(0, cycle_len - 1)
                edge2_idx = random.randint(0, cycle_len - 1)
                
                u1 = short_cycle[edge1_idx]
                v1 = short_cycle[(edge1_idx + 1) % cycle_len]
                u2 = short_cycle[edge2_idx]
                v2 = short_cycle[(edge2_idx + 1) % cycle_len]
                
                # Check that these edges are non-adjacent (no shared vertices)
                if u1 in (u2, v2) or v1 in (u2, v2):
                    continue
                
                # Check that (u1, u2) and (v1, v2) are not already edges
                if u2 in adjacency[u1] or v2 in adjacency[v1]:
                    continue
                
                # Check if switch is valid
                if _is_2_switch_valid(adjacency, u1, v1, u2, v2, g):
                    # Perform switch
                    adjacency[u1].remove(v1)
                    adjacency[v1].remove(u1)
                    adjacency[u2].remove(v2)
                    adjacency[v2].remove(u2)
                    adjacency[u1].add(u2)
                    adjacency[u2].add(u1)
                    adjacency[v1].add(v2)
                    adjacency[v2].add(v1)
                    break
    
    # Check final girth
    final_girth = _compute_girth(adjacency, n)
    if final_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def _greedy_construct(k, g, n, time_limit=10.0):
    # Try to construct a k-regular graph of girth >= g on n vertices
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    
    # Randomize the order of vertices to break symmetry
    vertices = list(range(n))
    random.shuffle(vertices)
    
    def has_short_cycle(u, v, max_depth):
        if u == v: return True
        visited = {u: 0}
        queue = [(u, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist >= max_depth:
                continue
            for neighbor in adjacency[node]:
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    active = list(range(n))
    random.shuffle(active)
    
    while active:
        if time.time() - start_time > time_limit:
            return None
            
        u_idx = random.randint(0, len(active) - 1)
        u = active[u_idx]
        
        found = False
        active_copy = active[:]
        random.shuffle(active_copy)
        
        for v in active_copy:
            if v == u:
                continue
            if v in adjacency[u]:
                continue
            
            if has_short_cycle(u, v, g - 2):
                continue
            
            adjacency[u].add(v)
            adjacency[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            found = True
            break
            
        if not found:
            return None
            
        new_active = []
        for node in active:
            if degrees[node] < k:
                new_active.append(node)
        active = new_active
        
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
            
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges

def construct(k, g):
    # For the specific case k=5, g=9, try switching algorithm
    if k == 5 and g == 9:
        # Time budget with safety margin (evaluator kills at 150s)
        total_time = 130.0
        start_time = time.time()
        
        # Try switching algorithm with various n values
        # Start with larger n where it's more feasible, then try smaller
        # The known best is 1116, so try around there and below
        target_ns = [800, 900, 1000, 1100, 1116, 1200, 1400, 1600, 1800, 2000]
        
        for n in target_ns:
            remaining = total_time - (time.time() - start_time)
            if remaining < 10:
                break
            # Give more time to smaller n values (harder problems)
            if n <= 1000:
                time_per_attempt = min(15.0, remaining / 3)
            else:
                time_per_attempt = min(10.0, remaining / 4)
            
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
                
    # Fallback to LUW
    return _luw_construct(k, g)
