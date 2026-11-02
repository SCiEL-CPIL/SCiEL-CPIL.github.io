import time
import random
from collections import deque

def _girth_nonbacktracking(adj, n, target):
    """
    Returns the girth if < target, else target (meaning girth >= target).
    Uses non-backtracking BFS from every vertex, but optimized.
    """
    best = target
    for s in range(n):
        # visited: dict from (node, prev) to depth
        visited = {}
        dq = deque()
        for v in adj[s]:
            visited[(s, v)] = 1
            dq.append((s, v, 1))
        while dq:
            u, prev, d = dq.popleft()
            if d + 1 >= best:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                if (v, u) in visited:
                    # Cycle found of length d+1
                    if d + 1 < best:
                        best = d + 1
                        if best < target:
                            return best
                else:
                    visited[(v, u)] = d + 1
                    if d + 1 < best:
                        dq.append((v, u, d + 1))
    return best

def _has_cycle_lt(adj, n, limit):
    """
    Returns True if there is a cycle of length < limit.
    Uses non-backtracking BFS.
    """
    for s in range(n):
        visited = set()
        dq = deque()
        for v in adj[s]:
            visited.add((s, v))
            dq.append((s, v, 1))
        while dq:
            u, prev, d = dq.popleft()
            if d + 1 >= limit:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                if (v, u) in visited:
                    return True
                else:
                    visited.add((v, u))
                    if d + 1 < limit:
                        dq.append((v, u, d + 1))
    return False

def _find_short_cycles(adj, n, max_len):
    """
    Returns a list of cycles of length <= max_len.
    Used for switching.
    """
    cycles = []
    # We look for cycles of length 4, 6, 8
    # For efficiency, we only find one of each length if possible, or a few.
    for s in range(n):
        # BFS storing parent
        parent = {}
        dist = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= max_len:
                continue
            for v in adj[u]:
                if v == s and dist[u] > 0:
                    L = dist[u] + 1
                    if L <= max_len:
                        # Reconstruct
                        path = [u]
                        curr = u
                        while curr != s:
                            curr = parent[curr]
                            path.append(curr)
                        path.reverse()
                        cycles.append(path)
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                else:
                    # Check if it forms a short cycle
                    # Simple check: if v is not parent of u and u is not parent of v
                    # (In undirected BFS, if v is visited, it's either parent or back/cross)
                    # If dist[u] + dist[v] + 1 <= max_len
                    # This logic is tricky for reconstruction without LCA.
                    # For now, we rely on the s-rooted back edges to s.
                    # This finds cycles containing s.
                    pass
    return cycles

def _construct_4_10_cage(deadline):
    n = 320
    best_adj = None
    best_girth = 0
    
    # Strategy:
    # 1. Try to find a 4-regular graph with girth >= 10 using local search.
    # 2. If not found, return best effort (which might fail validation if girth < 10, 
    #    but the task says "invalid result ... scores 0", so we MUST return a valid graph).
    #    Wait, the task says: "The graph must be simple ..., every vertex must have degree exactly k, 
    #    and no cycle may be shorter than g".
    #    So if we return a graph with girth 8, it is INVALID and scores 0.
    #    Therefore, we MUST find a valid graph or return a known valid one.
    #    The LUW construction is deterministic and valid (if it works).
    #    Does LUW(4, 10) work?
    #    The current code falls back to LUW.
    #    Let's check if LUW(4,10) produces girth >= 10.
    #    If LUW(4,10) is valid, we should just use it if local search fails.
    #    But maybe LUW(4,10) is not valid or is large?
    #    The task says "smallest known ... 320".
    #    LUW might produce > 320.
    #    If local search finds 320, great.
    #    If not, we return LUW result (which is valid, but maybe n > 320).
    #    Score is n. So 320 is better than LUW if LUW > 320.
    #    But if we can't find 320, we must return *something* valid.
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        adj = _generate_random_4_reg(n)
        if not adj:
            continue
            
        current_girth = _girth_nonbacktracking(adj, n, 10)
        
        if current_girth >= 10:
            return n, _adj_to_edges(adj)
            
        # Local search
        for step in range(2000):
            if time.time() > deadline - 2:
                break
                
            # Find a short cycle to break
            # We need a cycle of length < 10.
            # We can use _find_short_cycles but it's slow.
            # Alternative: Random edge swap.
            # Pick random edge (u,v). Pick random other edge (x,y) not incident.
            # Swap to (u,x) and (v,y) or (u,y) and (v,x).
            # Check if it creates multi-edges or loops.
            # Check if girth improves or stays same.
            
            u = random.randint(0, n-1)
            v = random.choice(adj[u])
            
            # Find a candidate edge to swap with
            swapped = False
            for _ in range(20):
                x = random.randint(0, n-1)
                if x == u or x == v:
                    continue
                y = random.choice(adj[x])
                if y == u or y == v:
                    continue
                if x == y:
                    continue
                # Ensure (u,x) and (v,y) are not existing edges
                if x in adj[u] or y in adj[v]:
                    continue
                # Ensure (u,y) and (v,x) are not existing edges? 
                # We are swapping (u,v) and (x,y) with (u,x) and (v,y).
                # So we remove (u,v) and (x,y).
                # We add (u,x) and (v,y).
                
                # Apply
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                new_girth = _girth_nonbacktracking(adj, n, 10)
                
                if new_girth >= 10:
                    return n, _adj_to_edges(adj)
                    
                if new_girth > current_girth:
                    current_girth = new_girth
                    swapped = True
                    break
                elif new_girth == current_girth:
                    if random.random() < 0.3:
                        swapped = True
                        break
                    else:
                        # Revert
                        adj[u].remove(x)
                        adj[x].remove(u)
                        adj[v].remove(y)
                        adj[y].remove(v)
                        
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[x].append(y)
                        adj[y].append(x)
                else:
                    # Revert
                    adj[u].remove(x)
                    adj[x].remove(u)
                    adj[v].remove(y)
                    adj[y].remove(v)
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[x].append(y)
                    adj[y].append(x)
                    
            if not swapped:
                # Try to find a specific short cycle and break it?
                # Or just restart local search?
                pass
                
        if best_adj is None or current_girth > best_girth:
            best_girth = current_girth
            best_adj = [lst[:] for lst in adj]
            
    # If we didn't find a valid graph, we must return a valid one.
    # The LUW construction is our fallback.
    # But wait, if best_adj has girth < 10, it's invalid.
    # We should return LUW.
    return None

def _generate_random_4_reg(n):
    if n % 2 != 0:
        return None
    stubs = [i for i in range(n) for _ in range(4)]
    random.shuffle(stubs)
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
    return adj

def _adj_to_edges(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _luw_construct(k, g):
    """Fallback LUW construction."""
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 4 and g == 10:
        try:
            result = _construct_4_10_cage(deadline)
            if result:
                n, edges = result
                # Verify
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                for i in range(n):
                    if len(adj[i]) != k:
                        raise ValueError
                girth = _girth_nonbacktracking(adj, n, g)
                if girth >= g:
                    return n, edges
        except Exception:
            pass
        
        # Fallback
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
