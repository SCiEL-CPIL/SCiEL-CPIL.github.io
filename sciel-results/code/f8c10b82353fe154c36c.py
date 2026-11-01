"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        # State: (vertex, prev_vertex), track min distance
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False  # found cycle shorter than g
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _construct_circulant(n, steps):
    """Construct a circulant graph given n and a list of steps.
    Returns (n, adj, edges) or None if invalid."""
    adj = [set() for _ in range(n)]
    edges = []
    for s in steps:
        if s <= 0 or s >= n:
            return None
        if s == n - s: # s = n/2, only one edge per vertex
            for i in range(n // 2):
                u = i
                v = i + s
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
        else:
            for i in range(n):
                u = i
                v = (i + s) % n
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return n, adj, edges

def _search_circulant(n, g, time_limit):
    """Search for a circulant graph of size n with girth >= g."""
    start_time = _time.time()
    # Valid steps are 1..n//2
    # We need 3 steps such that they are distinct and don't overlap in a way that creates multi-edges
    # Actually, if we pick steps s1, s2, s3, we must ensure s_i != s_j, s_i != n-s_j, etc.
    # And s_i != 0.
    
    # Initial random set
    def get_random_steps():
        steps = set()
        while len(steps) < 3:
            s = _random.randint(1, n // 2)
            # Ensure s is not the "negative" of an existing step
            # Since we only pick 1..n//2, s and n-s are the same edge set.
            # So we just need distinct values in 1..n//2.
            steps.add(s)
        return list(steps)

    best_steps = get_random_steps()
    
    # Local search
    while _time.time() < time_limit:
        result = _construct_circulant(n, best_steps)
        if result is None:
            best_steps = get_random_steps()
            continue
        
        n_val, adj, edges = result
        if _verify_girth(adj, n_val, g):
            return n_val, edges
        
        # Mutate: pick a random step and replace it
        idx = _random.randint(0, 2)
        old_step = best_steps[idx]
        new_step = _random.randint(1, n // 2)
        # Ensure new_step is not already in the set (other than the one being replaced)
        if new_step == old_step:
            continue
        if new_step in best_steps:
            continue
            
        best_steps[idx] = new_step

    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if k != 3:
        # Fallback to LUW for other k values
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
            add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # For k=3: Use circulant search
    # Moore bound for k=3, girth g: 2*(2^(g/2)-1) for even g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    # Search n from moore upwards
    # We will allocate time dynamically
    # Let's try a range of n values
    # Known best is 936. Let's search 512 to 1000.
    
    # To improve success rate, we can try multiple random restarts for each n.
    # Or, just iterate n and do local search.
    
    best_result = None
    
    # We will spend most time on n values close to the known optimum.
    # Let's prioritize n values that are multiples of small numbers or have good structure.
    # Actually, just linear scan is fine if local search is fast.
    
    # Allocate time per n:
    # If we have 100s, and we check 200 values of n, we have 0.5s per n.
    # This might be too short for local search to converge.
    
    # Strategy:
    # 1. Check n = moore, moore+2, ... up to moore + 100 with short time budget.
    # 2. Then check larger n with longer budget?
    # No, we want the SMALLEST n.
    
    # Let's do a binary-like search or just linear with increasing budget?
    # Linear is safer.
    
    n_current = moore
    if n_current % 2 != 0:
        n_current += 1
        
    # We will try up to n = 1000
    max_n = 1000
    
    while n_current <= max_n and _time.time() < deadline - 5:
        # Allocate time for this n
        # If we are early, give less time. If we are late, give more?
        # Let's give a fixed small amount, e.g., 0.2s, and increase if we fail?
        # Or just use remaining time / remaining n's.
        
        remaining_n = max_n - n_current + 1
        time_per_n = (deadline - 5 - _time.time()) / remaining_n
        # Cap time_per_n to avoid spending too much on one
        time_per_n = min(time_per_n, 2.0) 
        # Ensure at least 0.1s
        time_per_n = max(time_per_n, 0.1)
        
        res = _search_circulant(n_current, g, _time.time() + time_per_n)
        if res is not None:
            return res
            
        n_current += 2
        
    # Fallback: LUW construction with q=2 (if valid) or q=3
    # Try q=2 first
    if _time.time() < deadline - 5:
        q = 2
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        # Verify LUW result
        n_luw = len(order)
        adj = [set() for _ in range(n_luw)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check degree and girth
        valid = True
        for i in range(n_luw):
            if len(adj[i]) != k:
                valid = False
                break
        if valid:
            if _verify_girth(adj, n_luw, g):
                return n_luw, edges
        # If invalid or girth too low, try q=3
        if _time.time() < deadline - 5:
            q = 3
            add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
            rule = {2:(1,1), 3:(2,1), 4:(1,2)}
            for c in range(5, d+1):
                r, c0 = (c-5)%4, c-(c-5)%4
                rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
            def neighbours2(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0]*(d-1)
                    for c in range(2, d+1):
                        a, b = rule[c]
                        y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                    yield 1-side, tuple(y)
            start2 = (0, (None,) + (0,)*d)
            index2, order2, edges2 = {start2:0}, [start2], []
            for vertex in order2:
                for other in neighbours2(vertex):
                    if other not in index2:
                        index2[other] = len(order2)
                        order2.append(other)
                    if vertex[0] == 0:
                        edges2.append((index2[vertex], index2[other]))
            return len(order2), edges2

    # If all else fails, return the last tried circulant even if girth < g?
    # No, must be valid.
    # Return LUW q=3 result (which is valid but large)
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours3(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start3 = (0, (None,) + (0,)*d)
    index3, order3, edges3 = {start3:0}, [start3], []
    for vertex in order3:
        for other in neighbours3(vertex):
            if other not in index3:
                index3[other] = len(order3)
                order3.append(other)
            if vertex[0] == 0:
                edges3.append((index3[vertex], index3[other]))
    return len(order3), edges3
# EVOLVE-BLOCK-END
