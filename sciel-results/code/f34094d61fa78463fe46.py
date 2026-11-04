import time
import numpy as np
import math

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = np.repeat(np.arange(n, dtype=np.int64), k)
    perm = np.arange(m, dtype=np.int64)
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
        adj[u].append(v)
        adj[v].append(u)
        seen[u].add(v)
        seen[v].add(u)
    return adj

def _count_short_cycles(adj, g):
    """Exact count of cycles with length < g."""
    n = len(adj)
    count = 0
    max_depth = g - 1
    for s in range(n):
        dist = {}
        queue = []
        for v in adj[s]:
            dist[(v, s)] = 1
            queue.append((v, s))
        head = 0
        while head < len(queue):
            u, p = queue[head]
            head += 1
            d = dist[(u, p)]
            if d >= max_depth:
                continue
            for w in adj[u]:
                if w == p:
                    continue
                if w == s:
                    count += 1
                    continue
                key = (w, u)
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((w, u))
    return count

def _apply_2switch(adj, u, v, w, x):
    """Replace (u,v) and (w,x) with (u,w) and (v,x). Returns True if successful."""
    # Check validity
    if w in adj[u] or x in adj[v]:
        return False
    if u == w or u == x or v == w or v == x:
        return False
        
    # Remove u-v
    adj[u].remove(v)
    adj[v].remove(u)
    # Remove w-x
    adj[w].remove(x)
    adj[x].remove(w)
    # Add u-w
    adj[u].append(w)
    adj[w].append(u)
    # Add v-x
    adj[v].append(x)
    adj[x].append(v)
    return True

def _revert_2switch(adj, u, v, w, x):
    """Revert the switch: (u,w) and (v,x) back to (u,v) and (w,x)."""
    adj[u].remove(w)
    adj[w].remove(u)
    adj[v].remove(x)
    adj[x].remove(v)
    adj[u].append(v)
    adj[v].append(u)
    adj[w].append(x)
    adj[x].append(w)

def construct(k, g):
    deadline = time.time() + 115.0
    rng = 0x9E3779B97F4A7C15
    
    # Try to find a valid graph
    # Start with n=500
    n = 500
    adj = None
    while time.time() < deadline - 10.0:
        adj = _pairing_regular(n, k, rng)
        if adj is not None:
            break
        rng = _xorshift64(rng)
        
    if adj is None:
        # Fallback: return a simple valid 5-regular graph (e.g. cycle + chords)
        # But it won't have girth 9. Score 0.
        # Construct a simple 5-regular graph on 500 vertices
        adj = [[] for _ in range(n)]
        # Connect i to i+1, i+2, i+3, i-1, i-2 mod n? 
        # This creates 4-cycles.
        # Just return the first valid pairing if any, else empty
        # If no valid pairing found (unlikely), return empty
        return n, []

    # Initial cost
    current_cost = _count_short_cycles(adj, g)
    
    # Simulated Annealing
    T = 1.0
    best_adj = [lst[:] for lst in adj]
    best_cost = current_cost
    
    while time.time() < deadline:
        if current_cost == 0:
            best_adj = [lst[:] for lst in adj]
            best_cost = 0
            break
            
        # Pick random edge (u, v)
        u = _randrange(rng, n)
        v = adj[u][_randrange(rng, k)]
        
        # Pick random other edge (w, x)
        w = _randrange(rng, n)
        while w == u or w == v:
            w = _randrange(rng, n)
        x = adj[w][_randrange(rng, k)]
        if x == u or x == v:
            continue
        if (u == w and v == x) or (u == x and v == w):
            continue
        if w in adj[u] or x in adj[v]:
            continue
            
        # Try switch
        if _apply_2switch(adj, u, v, w, x):
            new_cost = _count_short_cycles(adj, g)
            delta = new_cost - current_cost
            if delta < 0 or (rng & 0xFFFFFFFF) < (0xFFFFFFFF * math.exp(-delta / T)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [lst[:] for lst in adj]
            else:
                _revert_2switch(adj, u, v, w, x)
        
        # Cool down
        T *= 0.99
        if T < 0.01:
            T = 0.01

    edges = [(i, j) for i in range(n) for j in best_adj[i] if i < j]
    return n, edges
