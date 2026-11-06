"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _common_neighbors_count(adj, a, b, c, n):
    """Count common neighbors of a, b, c (excluding a, b, c themselves)."""
    # adj is a list of sets
    common = adj[a] & adj[b] & adj[c]
    common.discard(a)
    common.discard(b)
    common.discard(c)
    return len(common)

def _is_k33_free(adj, n):
    """Check if graph is K3,3-free."""
    for a in range(n):
        for b in range(a+1, n):
            ab_common = adj[a] & adj[b]
            ab_common.discard(a)
            ab_common.discard(b)
            for c in range(b+1, n):
                # common neighbors of a, b, c
                cnt = 0
                for v in ab_common:
                    if v == c:
                        continue
                    if v in adj[c]:
                        cnt += 1
                        if cnt >= 3:
                            return False
    return True

def _count_violations(adj, n):
    """Count number of K3,3 violations (triples with >= 3 common neighbors)."""
    violations = 0
    for a in range(n):
        for b in range(a+1, n):
            ab_common = adj[a] & adj[b]
            ab_common.discard(a)
            ab_common.discard(b)
            for c in range(b+1, n):
                cnt = 0
                for v in ab_common:
                    if v == c:
                        continue
                    if v in adj[c]:
                        cnt += 1
                        if cnt >= 3:
                            violations += 1
                            break
    return violations

def _build_adj(edges, n):
    """Build adjacency list from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edges_from_adj(adj, n):
    """Build edge list from adjacency list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges

def _greedy_add_edges(adj, n, deadline):
    """Try to add edges greedily, maintaining K3,3-free property."""
    # Try all possible edges in random order
    all_edges = list(itertools.combinations(range(n), 2))
    random.shuffle(all_edges)
    
    added = 0
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if v in adj[u]:
            continue
        # Check if adding (u,v) creates a K3,3
        # A K3,3 would be created if there exist vertices a,b,c such that
        # a,b,c have >= 3 common neighbors after adding (u,v)
        # The new edge (u,v) can only create violations involving u or v
        # as part of the triple, or u/v as common neighbors
        
        # Check triples that include u or v
        # Case 1: triple includes u: {u, x, y}
        #   Common neighbors of u, x, y after adding: 
        #   Before: N(u) ∩ N(x) ∩ N(y)
        #   After: (N(u) ∪ {v}) ∩ N(x) ∩ N(y) = old + possibly v if v in N(x)∩N(y)
        #   So we need to check if v in N(x)∩N(y) and old count was 2
        # Case 2: triple includes v: similar
        # Case 3: triple doesn't include u or v: 
        #   Common neighbors change only if u or v is a common neighbor
        #   u is common neighbor of a,b,c if u in N(a)∩N(b)∩N(c)
        #   After adding (u,v), u's neighborhood changes, so this could matter
        
        # Simpler: just check all triples involving u or v
        ok = True
        # Triples with u
        for x in range(n):
            if x == u:
                continue
            for y in range(x+1, n):
                if y == u:
                    continue
                if x < y:
                    triple = (u, x, y)
                else:
                    triple = (u, y, x)
                # Count common neighbors after adding (u,v)
                # Common neighbors of u, x, y
                # N(u) after = N(u) ∪ {v}
                # So common = (N(u) ∪ {v}) ∩ N(x) ∩ N(y)
                common = adj[x] & adj[y]
                common.discard(u)
                common.discard(x)
                common.discard(y)
                # Add v if v is in N(x) and N(y)
                if v in adj[x] and v in adj[y]:
                    common.add(v)
                if len(common) >= 3:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        
        # Triples with v (but not u, already covered)
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                # Common neighbors of v, x, y after adding (u,v)
                # N(v) after = N(v) ∪ {u}
                common = adj[x] & adj[y]
                common.discard(v)
                common.discard(x)
                common.discard(y)
                # Add u if u is in N(x) and N(y)
                if u in adj[x] and u in adj[y]:
                    common.add(u)
                if len(common) >= 3:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        
        # Triples without u or v: check if u or v becomes a new common neighbor
        # u is a common neighbor of a,b,c if u ∈ N(a)∩N(b)∩N(c)
        # After adding (u,v), u's neighborhood gains v
        # So u can be a common neighbor of a,b,c only if v ∈ N(a)∩N(b)∩N(c)
        # Wait, that's not right. u being a common neighbor means u is adjacent to a, b, c.
        # Adding (u,v) doesn't change u's adjacency to a,b,c.
        # What changes is: v's adjacency gains u.
        # So v can become a common neighbor of a,b,c if v ∈ N(a)∩N(b)∩N(c) AND u ∈ N(a)∩N(b)∩N(c)
        # No wait: v is a common neighbor of a,b,c iff v is adjacent to a, b, c.
        # Adding (u,v) makes v adjacent to u. So v is a common neighbor of a,b,c 
        # iff v is adjacent to a, b, c. This doesn't change unless one of a,b,c is u.
        # If one of a,b,c is u, then v is adjacent to u (new), so v could be a common neighbor.
        # But we already checked triples with u.
        # Actually, for triple {a,b,c} not containing u or v:
        #   v is a common neighbor iff v ∈ N(a)∩N(b)∩N(c). This is unchanged by adding (u,v).
        #   u is a common neighbor iff u ∈ N(a)∩N(b)∩N(c). This is unchanged.
        # So no new violations for triples not containing u or v.
        
        # Actually wait - I need to double check. The common neighbors of a,b,c are 
        # vertices adjacent to all of a,b,c. Adding edge (u,v) changes:
        # - u's neighbors: gains v
        # - v's neighbors: gains u
        # So for triple {a,b,c}:
        # - If u ∈ {a,b,c}: u's neighbor set changes, so common neighbors might change
        # - If v ∈ {a,b,c}: v's neighbor set changes
        # - If u ∉ {a,b,c} and v ∉ {a,b,c}: 
        #   A vertex w is a common neighbor iff w adjacent to a,b,c
        #   Adding (u,v) only affects adjacency of u and v
        #   So if w=u: u is common neighbor iff u adj to a,b,c. Unchanged.
        #   If w=v: v is common neighbor iff v adj to a,b,c. Unchanged (v's adj to a,b,c unchanged)
        #   If w≠u,v: unchanged.
        # So indeed, only triples containing u or v need checking.
        
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
        added += 1
    
    return added

def _local_search(adj, n, deadline, max_iterations=5000):
    """Local search: try edge swaps to increase edge count while maintaining K3,3-free."""
    best_edges = len(_edges_from_adj(adj, n))
    improved = True
    iters = 0
    
    while improved and iters < max_iterations and time.time() < deadline:
        improved = False
        iters += 1
        
        # Try to add a random missing edge
        all_pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            
            # Check if adding (u,v) keeps graph K3,3-free
            # Only need to check triples containing u or v
            ok = True
            # Triples with u
            for x in range(n):
                if x == u:
                    continue
                for y in range(x+1, n):
                    if y == u:
                        continue
                    common = adj[x] & adj[y]
                    common.discard(u)
                    common.discard(x)
                    common.discard(y)
                    if v in adj[x] and v in adj[y]:
                        common.add(v)
                    if len(common) >= 3:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Triples with v (not u)
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(x+1, n):
                    if y == u or y == v:
                        continue
                    common = adj[x] & adj[y]
                    common.discard(v)
                    common.discard(x)
                    common.discard(y)
                    if u in adj[x] and u in adj[y]:
                        common.add(u)
                    if len(common) >= 3:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            best_edges += 1
            improved = True
            break
    
    return best_edges

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = None
    best_adj = None
    
    # Multiple random restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Start with random graph
        adj = [set() for _ in range(n)]
        all_pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        # Start with a moderately dense random graph
        p = 0.4 + 0.3 * random.random()
        for u, v in all_pairs:
            if random.random() < p:
                adj[u].add(v)
                adj[v].add(u)
        
        # Remove edges to make K3,3-free
        # Iteratively find and remove violating edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            all_pairs = list(itertools.combinations(range(n), 2))
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if v not in adj[u]:
                    continue
                # Check if removing (u,v) is needed: is (u,v) part of a violation?
                # Actually, let's check if the graph has any violation
                # Find a violating triple
                found_violation = False
                for a in range(n):
                    if found_violation:
                        break
                    for b in range(a+1, n):
                        if found_violation:
                            break
                        ab_common = adj[a] & adj[b]
                        ab_common.discard(a)
                        ab_common.discard(b)
                        for c in range(b+1, n):
                            cnt = 0
                            for w in ab_common:
                                if w == c:
                                    continue
                                if w in adj[c]:
                                    cnt += 1
                                    if cnt >= 3:
                                        # Found violation: triple (a,b,c)
                                        # Remove an edge. Which one?
                                        # Remove (u,v) if it's "involved"
                                        # Simplest: just remove (u,v) and see if it helps
                                        # Actually, let's just remove a random edge from the violation
                                        # The common neighbors are at least 3 vertices
                                        # Let's remove edge (a, w) where w is a common neighbor
                                        common_ws = []
                                        for w in ab_common:
                                            if w != c and w in adj[c]:
                                                common_ws.append(w)
                                        if common_ws:
                                            w = random.choice(common_ws)
                                            # Remove edge (a, w)
                                            adj[a].discard(w)
                                            adj[w].discard(a)
                                            changed = True
                                        found_violation = True
                                        break
                            if found_violation:
                                break
                        if found_violation:
                            break
                    if found_violation:
                        break
                if not found_violation:
                    break
        
        # Now adj is K3,3-free. Try to add more edges.
        _greedy_add_edges(adj, n, deadline)
        
        # Local search
        _local_search(adj, n, deadline)
        
        current_edges = len(_edges_from_adj(adj, n))
        if best_edges is None or current_edges > best_edges:
            best_edges = current_edges
            best_adj = [set(s) for s in adj]
    
    if best_adj is None:
        # Fallback to K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return _edges_from_adj(best_adj, n)

# EVOLVE-BLOCK-END
