<<<<<<< SEARCH
def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # girth check via non-backtracking BFS from each edge endpoint
    INF = 10 ** 9
    for (u, v) in edges:
        dist = {}
        q = []
        for w in adj[u]:
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v:
                best = L + 1
                break
            if L >= g - 1:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
=======
def _verify(n, edges, k, g):
    """Return True iff the graph is simple, k-regular, connected, and has girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check: for each vertex, do non-backtracking BFS and check
    # that no cycle of length < g is found.
    # A cycle of length L through vertex v means there's a non-backtracking
    # walk of length L-1 from one neighbor of v back to v (through another neighbor).
    # More precisely: for each vertex v, for each pair of distinct neighbors (a,b) of v,
    # check if there's a non-backtracking path from a to b of length < g-1 that doesn't use v.
    # Equivalently: run non-backtracking BFS from each directed edge (v->w) and check
    # if we can reach (v->u) for u != w within g-2 steps.
    
    # Efficient approach: for each vertex v, build the "non-backtracking" state space.
    # State is (prev, cur). Start from all (v, w) for w in adj[v].
    # We want to know if any (v, u) with u != w is reachable from (v, w) in < g-1 steps.
    # Actually, a cycle of length L containing edge (v,w) corresponds to a non-backtracking
    # walk of length L-1 from (v,w) back to (v,w) -- no, that's not right either.
    
    # Let me use the standard approach: for each directed edge (u,v), do BFS in the
    # non-backtracking graph. If we reach (v,u) at distance d, then there's a cycle
    # of length d+1. We need d+1 >= g, i.e., d >= g-1.
    # So we check that (v,u) is NOT reachable from (u,v) in fewer than g-1 steps.
    
    INF = 10 ** 9
    for (u, v) in edges:
        # BFS from directed edge (u,v) in non-backtracking graph
        # State: (prev, cur), meaning we arrived at cur from prev
        # Start: (u, v) at distance 0
        # Target: (v, u) -- if reached at distance d, cycle length is d+1
        # We need d+1 >= g, so d >= g-1
        # If we reach (v,u) at distance < g-1, fail.
        
        dist = { (u, v): 0 }
        queue = [(u, v)]
        head = 0
        found = False
        while head < len(queue):
            prev, cur = queue[head]
            head += 1
            d = dist[(prev, cur)]
            if d >= g - 1:
                break  # No need to go further; all cycles found are >= g
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    if (cur, w) == (v, u):
                        # Cycle of length d+1
                        if d + 1 < g:
                            return False
                        found = True
                        break
                    dist[(cur, w)] = d + 1
                    queue.append((cur, w))
            if found:
                break
        # If we didn't find (v,u) at all, that's fine (no cycle through this edge,
        # or the cycle is longer than we searched, which is >= g)
    return True
>>>>>>> REPLACE

Now let me improve the local search to be more structured and effective:

