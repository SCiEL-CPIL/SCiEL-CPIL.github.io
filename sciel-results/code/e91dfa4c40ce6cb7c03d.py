<<<<<<< SEARCH
def _embed(tree_edges, k, free, rng, node_limit=200000):
    """Find an injective host map for tree (vertices 0..k-1) using only free edges."""
    if k == 1:
        return [0]
    adj = [[] for _ in range(k)]
    for (u, v) in tree_edges:
        adj[u].append(v); adj[v].append(u)
    order = []
    parent = [-1] * k
    seen = [False] * k
    seen[0] = True
    dq = deque([0])
    while dq:
        x = dq.popleft()
        order.append(x)
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True; parent[y] = x; dq.append(y)
    host = [-1] * k
    used = set()
    counter = [0]

    def dfs(idx):
        if counter[0] > node_limit:
            return False
        if idx == len(order):
            return True
        v = order[idx]
        p = parent[v]
        if p == -1:
            cands = list(range(20))
        else:
            hp = host[p]
            cands = [h for h in range(20) if h != hp and frozenset((hp, h)) in free]
        rng.shuffle(cands)
        for h in cands:
            counter[0] += 1
            if counter[0] > node_limit:
                return False
            if h in used:
                continue
            host[v] = h
            used.add(h)
            if dfs(idx + 1):
                return True
            host[v] = -1
            used.remove(h)
        return False

    if dfs(0):
        return [int(x) for x in host]
    return None
=======
def _embed(tree_edges, k, free, rng, node_limit=500000):
    """Find an injective host map for tree (vertices 0..k-1) using only free edges."""
    if k == 1:
        return [0]
    adj = [[] for _ in range(k)]
    for (u, v) in tree_edges:
        adj[u].append(v); adj[v].append(u)
    order = []
    parent = [-1] * k
    children = [[] for _ in range(k)]
    seen = [False] * k
    seen[0] = True
    dq = deque([0])
    while dq:
        x = dq.popleft()
        order.append(x)
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True; parent[y] = x; children[x].append(y); dq.append(y)
    host = [-1] * k
    used = set()
    counter = [0]
    
    # Precompute free neighbors for fast access
    free_adj = [set() for _ in range(20)]
    for e in free:
        u, v = tuple(e)
        free_adj[u].add(v)
        free_adj[v].add(u)

    def dfs(idx):
        if counter[0] > node_limit:
            return False
        if idx == len(order):
            return True
        v = order[idx]
        p = parent[v]
        if p == -1:
            cands = list(range(20))
        else:
            hp = host[p]
            cands = list(free_adj[hp])
            
        if not cands:
            return False

        # MRV Heuristic: Sort candidates by the minimum number of available options for their children
        def score_candidate(h):
            if h in used:
                return -1
            if not children[v]:
                return 1
            min_opts = float('inf')
            for c in children[v]:
                # Candidates for child c are neighbors of h in free_adj, not in used
                opts = [x for x in free_adj[h] if x not in used]
                if not opts:
                    return -1
                if len(opts) < min_opts:
                    min_opts = len(opts)
            return min_opts

        scored_cands = []
        for h in cands:
            s = score_candidate(h)
            if s >= 0:
                scored_cands.append((s, h))
        
        if not scored_cands:
            return False
            
        # Sort by score (ascending - most constrained first), then randomize ties
        scored_cands.sort(key=lambda x: x[0])
        # Group by score to shuffle within groups
        groups = {}
        for s, h in scored_cands:
            if s not in groups:
                groups[s] = []
            groups[s].append(h)
        
        final_cands = []
        for s in sorted(groups.keys()):
            group = groups[s]
            rng.shuffle(group)
            final_cands.extend(group)

        for h in final_cands:
            counter[0] += 1
            if counter[0] > node_limit:
                return False
            host[v] = h
            used.add(h)
            if dfs(idx + 1):
                return True
            host[v] = -1
            used.remove(h)
        return False

    if dfs(0):
        return [int(x) for x in host]
    return None
>>>>>>> REPLACE

