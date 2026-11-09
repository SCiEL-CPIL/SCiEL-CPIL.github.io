"""Clique partition via density-prioritized greedy edge assignment + local improvement."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_LIMIT = 115.0


def _deadline_ok():
    return time.time() - _START < _LIMIT


def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _solve_one(n, p, seed):
    edges = _random_edges(n, p, seed)
    edge_set = set(edges)
    
    # Build adjacency as bitmasks for speed
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute common neighbors count for each edge (clique density indicator)
    # For edge (u,v), common neighbors = popcount(adj[u] & adj[v])
    def _popcount(x):
        return bin(x).count('1')
    
    # Greedy: repeatedly pick the edge with highest clique-density,
    # then grow a maximal clique around it, remove those edges.
    remaining = set(edges)
    chosen = []
    
    while remaining:
        if not _deadline_ok():
            break
        
        # Score each remaining edge by clique density
        best_edge = None
        best_score = -1
        for e in remaining:
            u, v = e
            common = _popcount(adj[u] & adj[v])
            # Higher common neighbors => can form larger clique
            if common > best_score:
                best_score = common
                best_edge = e
        
        if best_edge is None:
            break
        
        # Grow a clique starting from best_edge
        u0, v0 = best_edge
        clique = {u0, v0}
        # Candidates: common neighbors of all clique members
        candidates = adj[u0] & adj[v0]
        
        # Greedily add vertices with most connections to current clique
        while True:
            if not _deadline_ok():
                break
            # Find best candidate to add
            best_v = -1
            best_conn = 0
            cands_list = []
            # Iterate over set bits in candidates
            c = candidates
            while c:
                lb = c & (-c)
                v = lb.bit_length() - 1
                cands_list.append(v)
                c ^= lb
            
            for v in cands_list:
                # Count how many clique members are adjacent to v
                conn = 0
                for u in clique:
                    if adj[u] & (1 << v):
                        conn += 1
                if conn > best_conn:
                    best_conn = conn
                    best_v = v
            
            if best_v < 0:
                break
            
            clique.add(best_v)
            # Update candidates: must be adjacent to new vertex
            candidates &= adj[best_v]
        
        # Convert clique to edge list and remove from remaining
        clique_list = sorted(clique)
        cliq_edges = set()
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = clique_list[i], clique_list[j]
                if a > b:
                    a, b = b, a
                cliq_edges.add((a, b))
        
        # Only keep edges that are actually in the graph
        cliq_edges = cliq_edges & remaining
        
        if cliq_edges:
            chosen.append(clique_list)
            remaining -= cliq_edges
    
    # Fallback: any leftover edges as single-edge cliques
    for e in remaining:
        chosen.append([e[0], e[1]])
    
    # Local improvement: try to merge two cliques if their union is a clique
    if _deadline_ok():
        improved = True
        iters = 0
        while improved and iters < 30 and _deadline_ok():
            improved = False
            iters += 1
            new_chosen = list(chosen)
            for i in range(len(new_chosen)):
                if not _deadline_ok():
                    break
                for j in range(i + 1, len(new_chosen)):
                    if not _deadline_ok():
                        break
                    cl_i = set(new_chosen[i])
                    cl_j = set(new_chosen[j])
                    union = cl_i | cl_j
                    # Check if union is a clique
                    is_clique = True
                    for u in union:
                        for v in union:
                            if u != v and not (adj[u] & (1 << v)):
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique and len(union) >= 2:
                        new_chosen[i] = sorted(union)
                        new_chosen.pop(j)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                chosen = new_chosen
    
    return chosen


def construct():
    res1 = _solve_one(32, 0.6, 601)
    if not _deadline_ok():
        edges1 = _random_edges(32, 0.6, 601)
        res1 = [[e[0], e[1]] for e in edges1]
    
    res2 = _solve_one(64, 0.55, 602)
    if not _deadline_ok():
        edges2 = _random_edges(64, 0.55, 602)
        res2 = [[e[0], e[1]] for e in edges2]
    
    return [res1, res2]
# EVOLVE-BLOCK-END
