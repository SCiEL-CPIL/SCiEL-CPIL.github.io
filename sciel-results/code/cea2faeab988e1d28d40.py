import time
import numpy as np

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

def _find_short_cycle(adj, g):
    """Find one cycle of length < g. Returns list of vertices or None."""
    n = len(adj)
    max_depth = g - 1
    for s in range(n):
        # BFS non-backtracking
        # State: (current, parent)
        # dist[state] = length
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
                    # Found cycle of length d+1
                    # Reconstruct path? 
                    # For simplicity, we just return s, u, p... 
                    # Actually, we need the cycle edges to rewire.
                    # Let's return the path.
                    # Path: s -> ... -> p -> u -> s
                    # We need to reconstruct the path from s to u.
                    # This is expensive. 
                    # Alternative: Just return the start and end?
                    # No, we need to identify the cycle to break it.
                    # Let's store parent pointers.
                    pass
                key = (w, u)
                if key not in dist:
                    dist[key] = d + 1
                    queue.append((w, u))
    return None

# The above _find_short_cycle is incomplete. 
# Better approach: 
# 1. Count short cycles.
# 2. If > 0, pick a random edge and try to rewire it with a random other edge.
# 3. Check if count decreases.

def _count_short_cycles(adj, g):
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
    """Replace (u,v) and (w,x) with (u,w) and (v,x)."""
    # Remove u-v
    adj[u].remove(v)
    adj[v].remove(u)
    # Remove w-x
    adj[w].remove(x)
    adj[x].remove(w)
    # Add u-w
    if w not in adj[u]:
        adj[u].append(w)
        adj[w].append(u)
    else:
        # Invalid, revert
        adj[u].append(v)
        adj[v].append(u)
        adj[w].append(x)
        adj[x].append(w)
        return False
    # Add v-x
    if x not in adj[v]:
        adj[v].append(x)
        adj[x].append(v)
    else:
        # Invalid, revert
        adj[u].remove(w)
        adj[w].remove(u)
        adj[u].append(v)
        adj[v].append(u)
        adj[w].append(x)
        adj[x].append(w)
        return False
    return True

def construct(k, g):
    deadline = time.time() + 115.0
    rng = 0x9E3779B97F4A7C15
    
    # Target n
    # Moore bound for k=5, g=9 is 426.
    # We try n=500.
    n = 500
    
    adj = _pairing_regular(n, k, rng)
    if adj is None:
        return n, []
        
    current_cost = _count_short_cycles(adj, g)
    
    while time.time() < deadline:
        if current_cost == 0:
            break
        
        # Pick random edge (u, v)
        u = _randrange(rng, n)
        v = adj[u][_randrange(rng, k)]
        
        # Pick random other edge (w, x)
        w = _randrange(rng, n)
        # Ensure w != u, v
        while w == u or w == v:
            w = _randrange(rng, n)
        x = adj[w][_randrange(rng, k)]
        # Ensure x != u, v and edge (w,x) is not (u,v) or (v,u)
        # Also ensure (u,w) and (v,x) are not existing edges
        if x == u or x == v:
            continue
        if (u == w and v == x) or (u == x and v == w):
            continue
        if w in adj[u] or x in adj[v]:
            continue
            
        # Try switch
        old_adj = [lst[:] for lst in adj]
        if _apply_2switch(adj, u, v, w, x):
            new_cost = _count_short_cycles(adj, g)
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                # Revert
                adj = old_adj
        else:
            # Revert (already done in _apply_2switch if it returned False? 
            # No, _apply_2switch reverts internally if invalid. 
            # But we need to restore if we want to keep trying? 
            # Actually, if it returns False, adj is unchanged.
            pass

    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return n, edges
