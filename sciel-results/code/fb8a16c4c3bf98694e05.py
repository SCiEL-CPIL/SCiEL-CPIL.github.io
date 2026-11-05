"""Combined program: Fast LUW baseline with Randomized Local Search for smaller cages."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0


def _validate_graph(n, edges, k, g, max_vertices=None):
    """Validate simple k-regular graph with girth >= g."""
    if max_vertices is not None and n > max_vertices:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Girth check via non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return True


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if (g - 5) % 2 == 0:
        d = g - 4
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _count_short_cycles(adj, n, g):
    """Count cycles of length < g using non-backtracking BFS."""
    count = 0
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                if depth + 1 < g:
                    count += 1
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return count


def _random_4_regular(n, rng):
    """Generate a random 4-regular simple graph on n vertices using configuration model."""
    if n < 5:
        return None
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    rng.shuffle(stubs)
    
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Handle self-loops and multi-edges by retrying or swapping
    # Simple approach: build edge list, check validity, retry if invalid
    for attempt in range(20):
        edges = set()
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            # Check degree
            for i in range(n):
                if len(adj[i]) != 4:
                    valid = False
                    break
            if valid:
                return list(edges)
    return None


def _improve_girth_local_search(n, edges, g, time_limit, rng):
    """
    Use local search (edge swaps) to minimize the number of short cycles.
    Returns (edges, cost) if improved, else (None, None).
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    current_cost = _count_short_cycles(adj, n, g)
    if current_cost == 0:
        return edges, 0
        
    best_edges = list(edges)
    best_cost = current_cost
    
    t_start = time.time()
    # Temperature for simulated annealing
    temp = 10.0
    cooling = 0.99
    
    while time.time() < t_start + time_limit:
        if best_cost == 0:
            break
            
        # Pick a random edge to swap
        e1 = rng.choice(edges)
        u1, v1 = e1
        
        # Pick a random other edge
        e2 = rng.choice(edges)
        if e1 == e2:
            continue
        u2, v2 = e2
        
        # Try swapping: (u1,v1) and (u2,v2) -> (u1,u2) and (v1,v2)
        # Check if new edges are valid (no loops, no multi-edges)
        if u1 == u2 or v1 == v2 or u1 == v2 or v1 == u2:
            continue
        if u2 in adj[u1] or v2 in adj[v1]:
            continue
            
        # Apply swap
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        adj[u1].add(u2)
        adj[u2].add(u1)
        adj[v1].add(v2)
        adj[v2].add(v1)
        
        # Update edge list
        edges.remove(e1)
        edges.remove(e2)
        edges.append((min(u1, u2), max(u1, u2)))
        edges.append((min(v1, v2), max(v1, v2)))
        
        new_cost = _count_short_cycles(adj, n, g)
        
        # Accept if better or with probability exp(-(new_cost - current_cost)/temp)
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_edges = list(edges)
        else:
            # Simulated annealing acceptance
            if rng.random() < pow(sys.e, -(new_cost - current_cost) / temp):
                current_cost = new_cost
            else:
                # Reject swap, revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges.remove((min(u1, u2), max(u1, u2)))
                edges.remove((min(v1, v2), max(v1, v2)))
                edges.append(e1)
                edges.append(e2)
                
        temp *= cooling

    if best_cost < current_cost:
        return best_edges, best_cost
    return None, None


def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # LUW baseline
    best_n, best_edges = _luw_graph(k, g)
    
    if k != 4 or g != 10:
        return best_n, best_edges
    
    # Try to find a smaller graph using local search
    rng = random.Random(42)
    
    # Theoretical lower bound is 243. We know 320 exists.
    # Try sizes from 243 up to best_n - 1
    # We allocate time budget to each size
    total_search_time = deadline - time.time()
    if total_search_time <= 0:
        return best_n, best_edges
    
    # We will try a few random sizes first, then systematic
    # Prioritize sizes that are multiples of small numbers? No, just random.
    
    # To be efficient, we'll try a limited number of random restarts for each size
    # Start from the lower bound
    for n in range(243, best_n):
        if time.time() > deadline:
            break
        
        # Allocate a small time slice to each n
        # If we have many n's, we can't spend much time on each
        # Let's try 2 random restarts for each n
        found = False
        for attempt in range(2):
            if time.time() > deadline:
                break
            
            edges = _random_4_regular(n, rng)
            if edges is None:
                continue
            
            # Quick check: if it's already valid, return
            if _validate_graph(n, edges, k, g):
                best_n = n
                best_edges = edges
                found = True
                break
            
            # Try to improve
            time_slice = 2.0  # 2 seconds per attempt
            improved_edges, cost = _improve_girth_local_search(n, edges, g, time_slice, rng)
            if improved_edges is not None and cost == 0:
                if _validate_graph(n, improved_edges, k, g):
                    best_n = n
                    best_edges = improved_edges
                    found = True
                    break
        
        if found:
            # If we found a smaller one, we could stop, but we want the smallest.
            # Continue searching for even smaller ones?
            # Given time, we might not reach 243 if we start high.
            # We are iterating from 243 upwards, so the first one found is the smallest for this pass.
            # But we might find 300 first if we skipped 243-299.
            # We are iterating 243..319.
            # If we find 250, we break the loop? No, we should keep looking for < 250.
            # But we are going UP. So if we find 250, we have already tried 243-249.
            # So 250 is the best so far.
            # We can break if we find one, because any subsequent n is larger.
            break

    return best_n, best_edges

# EVOLVE-BLOCK-END
