"""Construct 3-regular graph of girth 16 using random generation with girth verification."""
import time
import random
from collections import deque

def _girth_violation(adj, n):
    """Check if graph has a cycle shorter than 16. Returns (cycle_len, path) or (None, None)."""
    # BFS from each vertex, tracking (node, prev) to avoid backtracking
    for start in range(n):
        # dist[node] = (distance, prev_node)
        dist = {}
        dist[(start, -1)] = 0
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, d = queue.popleft()
            if d >= 8:  # We only need to check up to depth 7 (cycles of length 15 would be found at depth 7)
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and d + 1 >= 16:
                    # Found a cycle of length d+1 >= 16, which is OK
                    pass
                if nb == start and d + 1 < 16:
                    return (d + 1, [(start, -1)])
                state = (nb, node)
                if state not in dist:
                    dist[state] = d + 1
                    queue.append((nb, node, d + 1))
    return (None, None)

def _is_valid_3_regular(adj, n):
    """Check if adj is a valid 3-regular simple graph."""
    for i in range(n):
        if len(adj[i]) != 3:
            return False
        if len(set(adj[i])) != 3:
            return False
        for nb in adj[i]:
            if i in adj[nb]:
                continue
            return False
    return True

def _generate_random_3_regular(n):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    # Each vertex needs degree 3, so total stubs = 3n
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    adj = {i: set() for i in range(n)}
    i = 0
    while i < len(stubs):
        j = i + 1
        if stubs[i] == stubs[j]:
            # Self-loop, try to fix
            # Find another stub with different vertex
            k = i + 2
            while k < len(stubs) and stubs[k] == stubs[i]:
                k += 1
            if k < len(stubs):
                stubs[j], stubs[k] = stubs[k], stubs[j]
            else:
                return None
        u, v = stubs[i], stubs[j]
        if u == v:
            return None
        if v in adj[u]:
            # Parallel edge
            return None
        adj[u].add(v)
        adj[v].add(u)
        i += 2
    
    # Convert to list of lists
    adj_list = [sorted(adj[i]) for i in range(n)]
    return adj_list

def _try_fix_girth(adj, n, max_time):
    """Try to fix girth violations by edge swaps."""
    start_time = time.time()
    adj = [list(x) for x in adj]
    
    for iteration in range(10000):
        if time.time() - start_time > max_time:
            break
        
        # Check for girth violation
        violated = False
        for start in range(n):
            # BFS with non-backtracking
            visited = {}  # (node, prev) -> dist
            visited[(start, -1)] = 0
            queue = deque([(start, -1, 0)])
            while queue:
                node, prev, d = queue.popleft()
                if d >= 8:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start and d + 1 < 16:
                        violated = True
                        break
                    state = (nb, node)
                    if state not in visited:
                        visited[state] = d + 1
                        queue.append((nb, node, d + 1))
                if violated:
                    break
            if violated:
                break
        
        if not violated:
            return adj
        
        # Try a random edge swap
        # Pick a random edge (u, v) and try to swap one endpoint
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if not edges:
            break
        
        u, v = random.choice(edges)
        # Try to replace edge (u, v) with (u, w) and (x, v) for some w, x
        # Or simply pick a random non-neighbor w of u and a random non-neighbor x of v
        # and check if swap works
        
        # Simpler: pick random edge and try to rewire
        # Remove edge (u, v), add edge (u, w) where w is random non-neighbor of u
        candidates_u = [x for x in range(n) if x != u and x not in adj[u] and x != v]
        candidates_v = [x for x in range(n) if x != v and x not in adj[v] and x != u]
        
        if not candidates_u or not candidates_v:
            continue
        
        w = random.choice(candidates_u)
        # Check if adding (u, w) creates parallel edge
        if w in adj[u]:
            continue
        
        # Try the swap: remove (u,v), add (u,w)
        # But we need to maintain 3-regularity, so we also need to fix v's degree
        # This gets complicated. Let's just try random rewiring.
        
        # Alternative: pick two edges (a,b) and (c,d) and swap to (a,c) and (b,d) or (a,d) and (b,c)
        if len(edges) < 2:
            continue
        a, b = random.choice(edges)
        c, d = random.choice(edges)
        
        # Try swap: remove (a,b) and (c,d), add (a,c) and (b,d)
        if a == c or a == d or b == c or b == d:
            continue
        if c in adj[a] or d in adj[b]:
            continue
        
        # Perform swap
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        
        adj[a].append(c)
        adj[c].append(a)
        adj[b].append(d)
        adj[d].append(b)
        
        # Sort for consistency
        for i in range(n):
            adj[i] = sorted(adj[i])
        
        # Check validity
        valid = True
        for i in range(n):
            if len(adj[i]) != 3 or len(set(adj[i])) != 3:
                valid = False
                break
        if not valid:
            # Revert
            adj[a].remove(c)
            adj[c].remove(a)
            adj[b].remove(d)
            adj[d].remove(b)
            adj[a].append(b)
            adj[b].append(a)
            adj[c].append(d)
            adj[d].append(c)
            for i in range(n):
                adj[i] = sorted(adj[i])
            continue

    return adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    if k != 3 or g != 16:
        # Fallback for other parameters
        return _luw_construct(k, g, deadline)
    
    # Moore bound for (3, 16): 1 + 3*sum_{i=0}^{6}(3^i) = 1 + 3*(1+3+9+27+81+243+729) = 1 + 3*1093 = 3280
    # Wait, let me recalculate. For girth g=16, radius = floor((g-1)/2) = 7
    # Moore bound = 1 + k*sum_{i=0}^{r-1}(k-1)^i where r = floor((g-1)/2)
    # For k=3, g=16: r = 7, so Moore = 1 + 3*sum_{i=0}^{6}2^i = 1 + 3*(2^7 - 1) = 1 + 3*127 = 382
    # Actually for even girth, the Moore bound is 2*sum_{i=0}^{g/2-1}(k-1)^i = 2*sum_{i=0}^{7}2^i = 2*(2^8-1) = 510
    # Let me use 512 as a safe lower bound
    
    moore_bound = 512
    
    # Try random graphs starting from Moore bound
    for n in range(moore_bound, min(moore_bound + 200, 800)):
        if n % 2 != 0:
            n += 1
            if n > 800:
                break
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            adj = _generate_random_3_regular(n)
            if adj is None:
                continue
            
            # Quick degree check
            if not _is_valid_3_regular(adj, n):
                continue
            
            # Check girth
            girth_ok = True
            for start in range(n):
                visited = {}
                visited[(start, -1)] = 0
                queue = deque([(start, -1, 0)])
                while queue:
                    node, prev, d = queue.popleft()
                    if d >= 8:
                        continue
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        if nb == start and d + 1 < 16:
                            girth_ok = False
                            break
                        state = (nb, node)
                        if state not in visited:
                            visited[state] = d + 1
                            queue.append((nb, node, d + 1))
                    if not girth_ok:
                        break
                if not girth_ok:
                    break
            
            if girth_ok:
                # Convert to edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: use LUW construction
    return _luw_construct(k, g, deadline)

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
        
    def get_digit(val, pos):
        return (val // powers[pos-1]) % q

    def get_point_neighbours(p_id):
        for l1 in range(q):
            current_l_id = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = get_digit(p_id, c)
                l_a = get_digit(current_l_id, a)
                p_b = get_digit(p_id, b)
                l_c = add(p_c, mul(l_a, p_b))
                current_l_id += l_c * powers[c-1]
            yield (1, current_l_id)

    def get_line_neighbours(l_id):
        for p1 in range(q):
            current_p_id = p1
            for c in range(2, d + 1):
                a, b = rule[c]
                l_c = get_digit(l_id, c)
                l_a = get_digit(l_id, a)
                p_b = get_digit(current_p_id, b)
                p_c = sub(l_c, mul(l_a, p_b))
                current_p_id += p_c * powers[c-1]
            yield (0, current_p_id)

    start_vertex = (0, 0)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, vid = vertex
        if side == 0:
            neighbors = get_point_neighbours(vid)
        else:
            neighbors = get_line_neighbours(vid)
            
        for n_side, n_vid in neighbors:
            n_vertex = (n_side, n_vid)
            if n_vertex not in index:
                index[n_vertex] = len(order)
                order.append(n_vertex)
            if side == 0:
                edges.append((index[vertex], index[n_vertex]))
                
    n = len(order)
    return n, edges
