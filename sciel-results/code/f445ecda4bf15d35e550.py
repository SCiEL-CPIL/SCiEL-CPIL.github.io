"""Rewritten program with faster C6 detection and stronger local search."""
import time
import random

# EVOLVE-BLOCK-START

def _creates_c6(adj, n, u, v):
    """Check if adding edge (u,v) to graph with adjacency sets creates a C6.
    
    A C6 is a simple cycle of length 6. Adding (u,v) creates a C6 iff
    there exists a simple path of length 5 from u to v in the current graph.
    Path: u - a - b - c - d - v with all vertices distinct.
    """
    Nu = adj[u]
    Nv = adj[v]
    
    # Enumerate: a in Nu, d in Nv, a != d, a != v, d != u
    # Then check if there's a path a-b-c-d of length 3 avoiding u,v
    for a in Nu:
        if a == v:
            continue
        Na = adj[a]
        for d in Nv:
            if d == u or d == a:
                continue
            Nd = adj[d]
            # Need b in Na \ {u,v}, c in N(b) \ {u,v,a} such that d in N(c)
            # i.e., d in N(c) means c in N(d)
            # So: exists b in Na\{u,v} such that (N(b) \ {u,v,a}) intersects N(d)
            for b in Na:
                if b == u or b == v or b == a:
                    continue
                Nb = adj[b]
                # Check if any c in Nb \ {u,v,a} is in Nd
                for c in Nb:
                    if c == u or c == v or c == a:
                        continue
                    if c in Nd:
                        return True
    return False

def _greedy_build(n, seed, deadline):
    """Build a C6-free graph greedily."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = set()
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    
    # Generate candidate edges sorted by heuristic
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                candidates.append((len(adj[u]) + len(adj[v]), u, v))
    
    candidates.sort(key=lambda x: (x[0], rng.random()))
    
    for _, u, v in candidates:
        if time.time() > deadline:
            break
        if not _creates_c6(adj, n, u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    return edges, adj

def _local_search(edges, adj, n, deadline, seed):
    """Local search with swap moves."""
    rng = random.Random(seed)
    best_count = len(edges)
    best_edges = set(edges)
    best_adj = [s.copy() for s in adj]
    
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to remove
        if not edges:
            break
        removed = rng.choice(tuple(edges))
        ru, rv = removed
        
        # Try to find a replacement
        added = None
        # Try random non-edges
        for _ in range(300):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                continue
            if (u, v) == removed:
                continue
            if not _creates_c6(adj, n, u, v):
                added = (u, v)
                break
        
        if added is None:
            no_improve += 1
            continue
        
        au, av = added
        
        # Perform swap
        edges.discard(removed)
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        edges.add(added)
        adj[au].add(av)
        adj[av].add(au)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
            best_adj = [s.copy() for s in adj]
            no_improve = 0
        else:
            # Revert
            edges.discard(added)
            adj[au].discard(av)
            adj[av].discard(au)
            edges.add(removed)
            adj[ru].add(rv)
            adj[rv].add(ru)
            no_improve += 1
    
    return best_edges, best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    n = 40
    best_overall = 0
    best_edges_overall = set()
    
    num_starts = 8
    time_per_start = (deadline - start_time) / num_starts
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        seed_start = 42 + s * 7919
        edges, adj = _greedy_build(n, seed_start, deadline - 5)
        
        if len(edges) > best_overall:
            best_overall = len(edges)
            best_edges_overall = set(edges)
        
        ls_deadline = min(deadline - 2, start_time + time_per_start * (s + 1))
        edges, adj, count = _local_search(edges, adj, n, ls_deadline, seed_start + 1)
        
        if count > best_overall:
            best_overall = count
            best_edges_overall = set(edges)
    
    # Final verification
    final_adj = [set() for _ in range(n)]
    for u, v in best_edges_overall:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    has_c6 = False
    for u in range(n):
        if has_c6:
            break
        for v in list(final_adj[u]):
            if v <= u:
                continue
            final_adj[u].discard(v)
            final_adj[v].discard(u)
            if _creates_c6(final_adj, n, u, v):
                has_c6 = True
                break
            final_adj[u].add(v)
            final_adj[v].add(u)
    
    if has_c6:
        fallback = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    fallback.add((u, v))
        return [list(e) for e in fallback]
    
    return [list(e) for e in best_edges_overall]

# EVOLVE-BLOCK-END
