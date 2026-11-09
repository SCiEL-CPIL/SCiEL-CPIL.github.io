import time
import random
from collections import deque

def _check_girth(adj, n, target_girth):
    """
    Check if the graph has girth at least target_girth.
    Uses BFS from each vertex.
    Returns the girth found.
    """
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
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
                    if min_girth < target_girth:
                        # Early exit optimization: if we just want to know if it's invalid
                        # we can return early. But we want the girth value for comparison.
                        # However, if min_girth is already < target_girth, 
                        # we can stop checking this start vertex? No, we need to check all starts?
                        # Actually, if min_girth < target_girth, the graph is invalid.
                        # We can return min_girth immediately if we only care about validity.
                        # But for local search, we want to know if it improved.
                        # So we continue.
                        pass
    return min_girth

def _generate_random_3_regular(n, seed):
    """
    Generate a random 3-regular graph on n vertices.
    Uses configuration model.
    """
    rng = random.Random(seed)
    if n % 2 != 0:
        return None
        
    stubs = []
    for i in range(n):
        stubs.extend([i] * 3)
    
    rng.shuffle(stubs)
    
    edges = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if (u, v) in edges or (v, u) in edges:
            return None
        edges.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    return n, list(edges), adj

def _local_search(n, target_girth, deadline, seed):
    """
    Local search to improve girth.
    """
    rng = random.Random(seed)
    
    attempts = 0
    while time.time() < deadline - 10:
        result = _generate_random_3_regular(n, rng.randint(0, 10**9))
        if result:
            break
        attempts += 1
        if attempts > 100:
            return None
            
    if not result:
        return None
        
    n, edges, adj = result
    girth = _check_girth(adj, n, target_girth)
    
    if girth >= target_girth:
        return n, edges
        
    best_girth = girth
    best_edges = list(edges)
    
    while time.time() < deadline - 5:
        if not edges:
            break
        e1 = edges[rng.randint(0, len(edges)-1)]
        e2 = edges[rng.randint(0, len(edges)-1)]
        
        u, v = e1
        x, y = e2
        
        if (u, v) == (x, y) or (u, v) == (y, x):
            continue
        if u in (x, y) or v in (x, y):
            continue
            
        # Try swap 1: (u,x) and (v,y)
        if u != x and v != y and (u, x) not in edges and (x, u) not in edges and (v, y) not in edges and (y, v) not in edges:
            edges.remove(e1)
            edges.remove(e2)
            edges.add((u, x))
            edges.add((v, y))
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            new_girth = _check_girth(adj, n, target_girth)
            
            if new_girth > best_girth:
                best_girth = new_girth
                best_edges = list(edges)
                
            if best_girth >= target_girth:
                return n, best_edges
                
            edges.remove((u, x))
            edges.remove((v, y))
            edges.add(e1)
            edges.add(e2)
            
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            
        # Try swap 2: (u,y) and (v,x)
        if u != y and v != x and (u, y) not in edges and (y, u) not in edges and (v, x) not in edges and (x, v) not in edges:
            edges.remove(e1)
            edges.remove(e2)
            edges.add((u, y))
            edges.add((v, x))
            
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            new_girth = _check_girth(adj, n, target_girth)
            
            if new_girth > best_girth:
                best_girth = new_girth
                best_edges = list(edges)
                
            if best_girth >= target_girth:
                return n, best_edges
                
            edges.remove((u, y))
            edges.remove((v, x))
            edges.add(e1)
            edges.add(e2)
            
            adj[u].remove(y)
            adj[y].remove(u)
            adj[v].remove(x)
            adj[x].remove(v)
            
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)

    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k == 3 and g == 16:
        # Try local search for increasing n
        for n in range(512, 4400, 32):
            if time.time() > deadline - 20:
                break
                
            for seed in range(10):
                if time.time() > deadline - 10:
                    break
                    
                result = _local_search(n, g, deadline, seed)
                if result:
                    return result
                    
        # Fallback to LUW
        q = 3
        d = 7
        
        def get_standard_rules(d):
            rule = {}
            if d >= 2: rule[2] = (1, 1)
            if d >= 3: rule[3] = (2, 1)
            if d >= 4: rule[4] = (1, 2)
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            return rule

        rules = get_standard_rules(d)
        
        def _build_luw(q, d, rules):
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: table[x][y]
            else:
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q

            start = (0, (None,) + (0,) * d)
            index = {start: 0}
            order = [start]
            edges = []
            
            for vertex in order:
                side, x = vertex
                for first in range(q):
                    y_list = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rules[c]
                        if side == 0:
                            y_list[c] = add(x[c], mul(y_list[a], x[b]))
                        else:
                            y_list[c] = sub(x[c], mul(x[a], y_list[b]))
                    
                    neighbor = (1 - side, tuple(y_list))
                    
                    if neighbor not in index:
                        index[neighbor] = len(order)
                        order.append(neighbor)
                    
                    if side == 0:
                        edges.append((index[vertex], index[neighbor]))
                        
            n = len(order)
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            for v in range(n):
                if len(adj[v]) != q:
                    return None
                    
            return n, edges, adj

        result = _build_luw(q, d, rules)
        if result:
            n, edges, adj = result
            if _check_girth(adj, n, g) >= g:
                return n, edges

    raise Exception("Failed to construct graph")
