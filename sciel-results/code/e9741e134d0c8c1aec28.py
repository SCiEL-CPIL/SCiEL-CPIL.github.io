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

def _generate_random_regular_graph(n, k, time_limit=5.0):
    """Generate a random k-regular graph using configuration model + 2-switches to make simple."""
    start_time = time.time()
    if n * k % 2 != 0:
        return None
    
    # Configuration model
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    # Shuffle
    random.shuffle(half_edges)
    
    # Pair up
    edges = []
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i+1]
        if u != v:
            edges.append((u, v))
    
    # Build adjacency and check for multi-edges
    adj = [set() for _ in range(n)]
    valid = True
    for u, v in edges:
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
    
    if not valid:
        # Try 2-switches to fix multi-edges
        for _ in range(10000):
            if time.time() - start_time > time_limit:
                return None
            # Find a multi-edge
            multi = None
            for u in range(n):
                for v in adj[u]:
                    if u < v and len(adj[u] & adj[v]) > 0:
                        # This is just checking if there's a multi-edge, not quite right
                        pass
            # Simpler: rebuild from scratch
            return None
    
    return adj

def _check_girth(adj, n, g, time_limit=10.0):
    """Check if graph has girth >= g using non-backtracking walks from each vertex."""
    start_time = time.time()
    
    # For each vertex, do BFS with non-backtracking constraint
    # A non-backtracking walk is a walk where you don't immediately go back to where you came from
    # We check if there's a cycle of length < g
    
    for start in range(n):
        # BFS tracking (node, prev_node, depth)
        # We want to find if we can return to start with a non-backtracking walk of length < g
        # Actually, we need to check all cycles, not just those through start
        # Better: for each vertex, do BFS and check if we can reach any neighbor within distance < g-1
        # but with non-backtracking constraint
        
        # Simpler approach: check all pairs (u,v) where dist(u,v) < g-1
        # but that's O(n^2)
        
        # For now, just do a basic BFS from each vertex to check for short cycles
        # This is O(n * (n+k)) which might be too slow for n=1006
        if time.time() - start_time > time_limit:
            return None
        
        visited = {}
        queue = deque()
        for neighbor in adj[start]:
            queue.append((neighbor, start, 1))
            visited[(neighbor, start)] = 1
        
        while queue:
            if time.time() - start_time > time_limit:
                return None
            node, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for neighbor in adj[node]:
                if neighbor == prev:
                    continue  # non-backtracking
                if neighbor == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                if (neighbor, node) not in visited:
                    visited[(neighbor, node)] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    
    return True

def _girth_switching_construct(k, g, n, time_limit=80.0):
    """Try to construct a k-regular graph of girth >= g using edge-switching."""
    start_time = time.time()
    
    # Generate initial random k-regular graph
    adj = None
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            return None
        adj = _generate_random_regular_graph(n, k, time_limit=2.0)
        if adj is not None:
            break
    
    if adj is None:
        return None
    
    # Check initial girth
    if _check_girth(adj, n, g, time_limit=5.0) == True:
        # Already good
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Edge-switching: pick two edges (a,b) and (c,d), replace with (a,c) and (b,d) or (a,d) and (b,c)
    # Accept if girth increases or stays the same
    
    # Build edge list for easy switching
    edge_list = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edge_list.append((i, j))
    
    current_girth = g  # We'll track this approximately
    
    for step in range(500000):
        if time.time() - start_time > time_limit:
            return None
        
        # Pick two random edges
        idx1 = random.randint(0, len(edge_list) - 1)
        idx2 = random.randint(0, len(edge_list) - 1)
        if idx1 == idx2:
            continue
        
        a, b = edge_list[idx1]
        c, d = edge_list[idx2]
        
        # Check if these edges share a vertex
        if a == c or a == d or b == c or b == d:
            continue
        
        # Try switch: remove (a,b) and (c,d), add (a,c) and (b,d)
        # Check if this creates multi-edges
        if c in adj[a] or d in adj[b]:
            continue
        
        # Perform the switch
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        adj[a].add(c)
        adj[c].add(a)
        adj[b].add(d)
        adj[d].add(b)
        
        edge_list[idx1] = (a, c)
        edge_list[idx2] = (b, d)
        
        # Check if girth is still >= g (local check)
        # Only check neighborhoods of a, b, c, d
        # This is an approximation; we'll do a full check periodically
        if step % 1000 == 0:
            girth_ok = _check_girth(adj, n, g, time_limit=3.0)
            if girth_ok == True:
                # We found a good graph
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
            elif girth_ok == None:
                # Time limit, keep going
                pass
            else:
                # Girth decreased, revert the switch
                adj[a].remove(c)
                adj[c].remove(a)
                adj[b].remove(d)
                adj[d].remove(b)
                adj[a].add(b)
                adj[b].add(a)
                adj[c].add(d)
                adj[d].add(c)
                edge_list[idx1] = (a, b)
                edge_list[idx2] = (c, d)
    
    return None

def construct(k, g):
    # For the specific case k=5, g=9, try edge-switching
    if k == 5 and g == 9:
        # Moore bound for k=5, g=9:
        # For odd girth, Moore bound is k * (k-1)^(g/2) / 2 + ... 
        # Actually, for cage problems, the lower bound is around 428
        # Best known is 1116
        # Try sizes near the known best
        target_ns = [1116, 1120, 1150, 1200, 1300, 1500]
        
        time_budget = 100.0
        time_per_attempt = time_budget / len(target_ns)
        
        for n in target_ns:
            res = _girth_switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
    # Fallback to LUW
    return _luw_construct(k, g)
