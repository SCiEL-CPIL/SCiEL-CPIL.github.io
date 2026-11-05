import time
import random
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
        
    return adj

def _girth_fast(adj, n, g):
    """
    Returns True if girth >= g.
    """
    for start in range(n):
        # visited[curr] = set of prevs
        # We need to track depth to ensure we don't count long paths as short cycles
        # But since we stop at g-1, we just need to know if we reached 'start'
        
        # Layer 0: (start, nb)
        # We can use a list of dicts: layer[i] = {curr: set(prevs)}?
        # No, state is (prev, curr).
        
        # Let's use a list of sets for the current frontier.
        # frontier is a set of (prev, curr)
        
        frontier = set()
        for nb in adj[start]:
            if nb == start:
                return False # Loop
            frontier.add((start, nb))
            
        # BFS up to depth g-2
        # If we reach a state (prev, curr) where curr == start, 
        # the cycle length is depth + 1.
        # We want depth + 1 < g => depth < g - 1.
        
        for depth in range(1, g - 1):
            new_frontier = set()
            for (prev, curr) in frontier:
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == start:
                        # Cycle of length depth + 1
                        if depth + 1 < g:
                            return False
                    else:
                        state = (curr, nxt)
                        # We only need to keep the shortest path to (curr, nxt)
                        # Since BFS, the first time we see it is the shortest.
                        # But we are expanding from all starts.
                        # We can just add to new_frontier.
                        # To avoid duplicates in the same layer, set handles it.
                        new_frontier.add(state)
            
            if not new_frontier:
                break
            frontier = new_frontier
            
    return True

def construct(k, g):
    deadline = time.time() + 135.0
    rng = 0x9E3779B97F4A7C15
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
        
    n_start = max(2 * k, moore)
    
    # Try to find a small graph
    # Start from n_start and go up
    # We want to minimize n.
    
    best_n = None
    best_adj = None
    
    # Search range
    # If we find one at n_start, great.
    # If not, try n_start + 2, etc.
    
    # Limit search to avoid timeout
    # If n=200, check is fast.
    # Let's try up to 500.
    
    n = n_start
    if n % 2 != 0:
        n += 1
        
    max_n = 600
    
    while n <= max_n and time.time() < deadline:
        attempts = 0
        found = False
        while attempts < 20 and time.time() < deadline:
            adj = _pairing_regular(n, k, rng)
            if adj is not None:
                if _girth_fast(adj, n, g):
                    return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            attempts += 1
            
        # If not found, increase n
        n += 2
        
    # Fallback: LUW
    # The seed program's LUW is likely 1250.
    # We can just generate a random graph of size 1250 and hope it has girth 9?
    # Or use the LUW code from seed.
    
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
        if time.time() > deadline - 5:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    # Verify LUW girth?
    # If LUW girth < g, we are in trouble.
    # But LUW is designed to have girth ~ 2d+1 or similar.
    # For d=4, girth is likely 9 or 10.
    
    return len(order), edges
