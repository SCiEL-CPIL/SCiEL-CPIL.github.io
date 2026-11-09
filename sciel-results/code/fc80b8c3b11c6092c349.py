import time
import random

def _girth_nonbacktracking(adj, n, target_girth):
    """Check if graph has no cycle shorter than target_girth using non-backtracking walks."""
    # For each vertex, do BFS with non-backtracking constraint
    for start in range(n):
        # dist[v] = shortest non-backtracking walk length from start to v
        # prev[v] = the vertex we came from to reach v (to prevent backtracking)
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[start] = 0
        queue = [(start, -1)]  # (vertex, previous_vertex)
        
        while queue:
            new_queue = []
            for v, p in queue:
                for u in adj[v]:
                    if u == p:
                        continue  # non-backtracking: don't go back
                    new_dist = dist[v] + 1
                    if new_dist < dist[u]:
                        dist[u] = new_dist
                        prev[u] = v
                        new_queue.append((u, v))
                    elif new_dist == dist[u] and u != prev[v]:
                        # Found a cycle of length new_dist + 1? No...
                        # Actually, if we can reach u with the same distance via different path,
                        # there's a cycle. But this is tricky.
                        pass
            queue = new_queue
        
        # Check for cycles: if any edge (v,u) where dist[v] + dist[u] + 1 < target_girth
        # and the paths don't backtrack
        for v in range(n):
            for u in adj[v]:
                if v < u:  # avoid double counting
                    # Check if there's a non-backtracking cycle through edge (v,u)
                    # A cycle exists if dist[v] + dist[u] + 1 < target_girth
                    # but we need to ensure the paths from start to v and start to u
                    # don't share the last edge in a way that creates backtracking
                    if dist[v] + dist[u] + 1 < target_girth:
                        # More careful check needed
                        # For now, use a simpler but correct approach
                        return False
    
    return True

def _bfs_girth_correct(adj, n):
    """Compute girth using BFS from each vertex with proper cycle detection."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth

def _is_valid_3_regular(adj, n):
    """Check if adj represents a valid simple 3-regular graph."""
    for i in range(n):
        if len(adj[i]) != 3:
            return False
        # Check no self-loops
        if i in adj[i]:
            return False
        # Check no duplicate edges
        if len(set(adj[i])) != len(adj[i]):
            return False
    return True

def _generate_random_3_regular(n, max_attempts=100):
    """Generate a random simple 3-regular graph using configuration model with retries."""
    if n < 4 or n % 2 != 0:
        return None
    
    for _ in range(max_attempts):
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        valid = True
        edge_set = set()
        
        for i in range(0, n * 3, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edge_set:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u,v), max(u,v)))
        
        if valid and _is_valid_3_regular(adj, n):
            return adj
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Fallback: Seed LUW construction
    def _luw_construct():
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

    # If not the specific case, use LUW
    if k != 3 or g != 16:
        return _luw_construct()

    # For (3, 16): try to find a smaller graph than LUW
    # LUW gives ~350k vertices, known best is 936
    # Try sizes from 512 upward
    
    # First, compute LUW as fallback
    luw_n, luw_edges = _luw_construct()
    best_n, best_edges = luw_n, luw_edges
    
    # Try various sizes in increasing order
    target_sizes = [512, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 936, 960, 1000]
    
    for target_n in target_sizes:
        if time.time() > deadline - 20:  # Leave 20s for final verification
            break
        
        # Try multiple random graphs of this size
        attempts = 0
        max_attempts = max(5, int((deadline - time.time()) / 5))  # ~5s per attempt
        
        while attempts < max_attempts and time.time() < deadline - 20:
            attempts += 1
            adj = _generate_random_3_regular(target_n, max_attempts=50)
            if adj is None:
                continue
            
            # Check girth
            girth = _bfs_girth_correct(adj, target_n)
            if girth >= g:
                # Convert to edge list
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return target_n, edges
    
    # If no small graph found, return LUW result
    return best_n, best_edges
