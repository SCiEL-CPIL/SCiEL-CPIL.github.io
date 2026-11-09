import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 3 and g == 16:
        # Attempt to construct a smaller graph via local search
        # The Moore bound for k=3, g=16 is 510.
        # Known smallest is 936.
        
        for n in range(512, 1000, 8):
            if time.time() > deadline - 10:
                break
            
            # Try a few random starts for this size
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                
                edges = _generate_random_3_reg(n)
                if not edges:
                    continue
                
                # Check if girth is already sufficient using non-backtracking walks
                if _check_girth_nb(edges, n, g):
                    return n, edges
                
                # Local search to improve girth
                edges = _local_search_girth(edges, n, g, deadline)
                
                if _check_girth_nb(edges, n, g):
                    return n, edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    return _luw_construct(k, g)

def _generate_random_3_reg(n):
    if n < 4 or n % 2 != 0:
        return None
    if n * 3 % 2 != 0:
        return None
        
    adj = [[] for _ in range(n)]
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self loop, invalid
            return None
        if v in adj[u]:
            # Multi edge, invalid
            return None
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    # Final check
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return edges

def _check_girth_nb(edges, n, g):
    """Check girth >= g using non-backtracking walks.
    
    A graph has girth >= g if and only if there is no non-backtracking
    closed walk of length < g. We check this by, for each vertex, doing
    a DFS of non-backtracking walks up to depth g-1 and checking if we
    return to the start.
    
    For efficiency, we use the fact that in a 3-regular graph, the number
    of non-backtracking walks of length L from a vertex is at most 3 * 2^(L-1).
    For g=16, that's 3 * 2^15 = 98304 per vertex, which is feasible for
    moderate n.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, check if there's a non-backtracking closed walk of length < g
    # We use BFS on the state (current_vertex, previous_vertex, length)
    # but limit depth to g-1
    
    for start in range(n):
        # State: (current, prev, length)
        # We want to know if we can return to 'start' with length < g
        # and the last step doesn't backtrack (i.e., we don't go back to prev)
        
        # Use a set to avoid revisiting states
        # State is (current, prev) where prev is the vertex we came from
        # We track the minimum length to reach each state
        
        visited = set()
        queue = deque()
        
        # Start: from 'start', we can go to any neighbor with prev = start
        for nb in adj[start]:
            state = (nb, start)
            visited.add(state)
            queue.append((nb, start, 1))
        
        while queue:
            curr, prev, length = queue.popleft()
            
            # Check if we've returned to start
            if curr == start and length < g:
                return False
            
            # Don't expand beyond depth g-1
            if length >= g - 1:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue  # Non-backtracking: can't go back to where we came from
                state = (nb, curr)
                if state not in visited:
                    visited.add(state)
                    queue.append((nb, curr, length + 1))
    
    return True

def _local_search_girth(edges, n, g, deadline):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for iteration in range(200):
        if time.time() > deadline - 2:
            break
            
        # Find a short cycle using non-backtracking walk detection
        min_len, cycle_edge = _find_short_cycle_nb(adj, n, g)
        if min_len >= g:
            break
            
        if cycle_edge is None:
            break
            
        u, v = cycle_edge
        
        # Try to break it by swapping edges
        found = False
        for _ in range(10):
            x, y = _random_non_edge_pair(adj, n)
            if x is None:
                continue
            # Check validity of swap (u,v) with (x,y)
            if u == x or u == y or v == x or v == y:
                continue
            if x in adj[u] or y in adj[v]:
                continue
            if x in adj[v] or y in adj[u]:
                continue
            # Also ensure (u,x) and (v,y) are not existing edges
            if x in adj[u] or y in adj[v]:
                continue
            if u in adj[x] or v in adj[y]:
                continue
                
            # Perform swap
            new_edges = _swap_edges(edges, (u, v), (x, y))
            if new_edges is None:
                continue
                
            new_adj = [[] for _ in range(n)]
            for a, b in new_edges:
                new_adj[a].append(b)
                new_adj[b].append(a)
                
            # Check if girth improved
            new_min_len, _ = _find_short_cycle_nb(new_adj, n, g)
            if new_min_len >= min_len:
                edges = new_edges
                adj = new_adj
                found = True
                break
                
        if not found:
            # Try a different approach: random edge swap
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                # Pick two random edges and try to swap
                if len(edges) < 2:
                    break
                e1_idx, e2_idx = random.sample(range(len(edges)), 2)
                e1, e2 = edges[e1_idx], edges[e2_idx]
                u, v = e1
                x, y = e2
                if u == x or u == y or v == x or v == y:
                    continue
                if x in adj[u] or y in adj[v] or x in adj[v] or y in adj[u]:
                    continue
                if u in adj[x] or v in adj[y]:
                    continue
                    
                new_edges = _swap_edges(edges, e1, e2)
                if new_edges is None:
                    continue
                    
                new_adj = [[] for _ in range(n)]
                for a, b in new_edges:
                    new_adj[a].append(b)
                    new_adj[b].append(a)
                    
                new_min_len, _ = _find_short_cycle_nb(new_adj, n, g)
                if new_min_len >= min_len:
                    edges = new_edges
                    adj = new_adj
                    found = True
                    break
    
    return edges

def _find_short_cycle_nb(adj, n, g):
    """Find the shortest cycle length using non-backtracking walks.
    
    Returns (min_cycle_length, (u, v)) where (u,v) is an edge on a shortest
    cycle, or (g, None) if no cycle shorter than g is found.
    """
    min_len = g
    best_edge = None
    
    for start in range(n):
        # BFS on non-backtracking walk states
        # State: (current, prev, length)
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            state = (nb, start)
            visited.add(state)
            queue.append((nb, start, 1))
        
        while queue:
            curr, prev, length = queue.popleft()
            
            if curr == start and length < min_len:
                min_len = length
                best_edge = (prev, start)  # The edge that closes the cycle
                break
            
            if length >= g - 1:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in visited:
                    visited.add(state)
                    queue.append((nb, curr, length + 1))
        
        if min_len < g:
            break
    
    return min_len, best_edge

def _random_non_edge_pair(adj, n):
    for _ in range(50):
        u, v = random.sample(range(n), 2)
        if v not in adj[u]:
            return u, v
    return None, None

def _swap_edges(edges, e1, e2):
    u, v = e1
    x, y = e2
    new_edges = []
    for e in edges:
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            continue
        if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
            continue
        new_edges.append(e)
    new_edges.append((u, x))
    new_edges.append((v, y))
    return new_edges

def _luw_construct(k, g):
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
