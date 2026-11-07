"""Clique partition of split graphs via greedy absorption + Max-Clique cover."""
import time
from random import Random


def construct():
    _T0 = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _cc_edge(a, b):
        return (a, b) if a < b else (b, a)

    def _max_clique(adj, n):
        """Find a maximum clique in a graph with n vertices and adjacency list adj.
        Uses Bron-Kerbosch with pivot.
        Returns the clique as a set of vertices.
        """
        best_clique = set()
        
        def bronk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return
            
            # Choose pivot u from P U X
            # Heuristic: pick u with max degree in P
            pivot = None
            max_deg = -1
            for u in P:
                deg = len(adj[u] & P)
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # If P is empty, pivot can be from X or None
            if not P:
                if X:
                    # Pick any from X? Or just return if X not empty?
                    # Standard BK: if P is empty, we are done with this branch if we update best.
                    # Actually, if P is empty, no more vertices can be added.
                    # The condition `if not P and not X` handles the leaf.
                    # If P is empty but X is not, we still check best_clique.
                    if len(R) > len(best_clique):
                        best_clique = set(R)
                    return

            # Iterate over P \ N(pivot)
            candidates = list(P - adj[pivot]) if pivot is not None else list(P)
            for v in candidates:
                # Add v to R
                R.add(v)
                # Recurse
                bronk(R, P & adj[v], X & adj[v])
                # Remove v from R
                R.discard(v)
                # Move v from P to X
                P.discard(v)
                X.add(v)
                
                # Early termination? 
                # If len(R) + len(P) <= len(best_clique), we can prune.
                if len(R) + len(P) <= len(best_clique):
                    # We can't beat best_clique in this branch
                    # But we still need to move remaining P to X? 
                    # Actually, if we prune, we just return.
                    # But we need to restore state? 
                    # In this implementation, P and X are passed by reference (mutated).
                    # This is tricky. 
                    # Let's stick to simple recursion without aggressive pruning for n=30.
                    pass

        bronk(set(), set(range(n)), set())
        return best_clique

    def _cover_remaining_cc(remaining, k):
        """Cover the remaining C-C edges with C-only cliques.
        Uses Maximum Clique extraction.
        """
        if not remaining:
            return []
        
        cliques = []
        remaining = set(remaining)
        
        # We need to rebuild adjacency for the current remaining graph
        # Vertices are 0..k-1
        while remaining:
            # Build adjacency for current remaining edges
            adj = [set() for _ in range(k)]
            active = set()
            for (a, b) in remaining:
                adj[a].add(b)
                adj[b].add(a)
                active.add(a)
                active.add(b)
            
            if not active:
                break

            # Map active vertices to 0..m-1 for the BK algorithm?
            # Or just run BK on the full graph but restricting to active?
            # BK on full graph with P=active is fine.
            
            # Create a dummy graph structure for BK?
            # BK needs adj lists. 
            # We have adj for all k vertices.
            
            # Run Max Clique on the subgraph induced by 'active'
            # We can just call _max_clique with a wrapper or modify it.
            # Let's create a local adjacency list for active vertices only to be safe and fast.
            
            active_list = list(active)
            idx_map = {v: i for i, v in enumerate(active_list)}
            m = len(active_list)
            local_adj = [set() for _ in range(m)]
            
            for (a, b) in remaining:
                if a in idx_map and b in idx_map:
                    ia = idx_map[a]
                    ib = idx_map[b]
                    local_adj[ia].add(ib)
                    local_adj[ib].add(ia)
            
            max_cl = _max_clique(local_adj, m)
            
            if not max_cl:
                # Should not happen if remaining is not empty
                break
            
            # Map back to original vertex ids
            clique_vertices = {active_list[i] for i in max_cl}
            
            # Remove edges within this clique from remaining
            cl_sorted = sorted(clique_vertices)
            for i in range(len(cl_sorted)):
                for j in range(i + 1, len(cl_sorted)):
                    remaining.discard(_cc_edge(cl_sorted[i], cl_sorted[j]))
            
            cliques.append(cl_sorted)
            
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add(_cc_edge(a, b))

        # N_C[v] = neighbors of I-vertex v in C
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if _cc_edge(u, v) in edge_set)

        # all C-C edges
        all_cc = set(_cc_edge(u, v) for u in range(k) for v in range(u + 1, k))
        cc_list = list(all_cc)

        # Build bipartite graph: left = C-C edges, right = I-vertices
        # Edge if I-vertex is adjacent to both endpoints of C-C edge
        # Use Hopcroft-Karp or simple greedy matching
        
        # For each C-C edge (u,v), find I-vertices adjacent to both u and v
        cc_to_iverts = {}
        for idx, (u, v) in enumerate(cc_list):
            candidates = []
            for w in range(k, n):
                if u in N_C[w] and v in N_C[w]:
                    candidates.append(w)
            cc_to_iverts[idx] = candidates

        # Greedy matching: match C-C edges to I-vertices
        # Each I-vertex can be used multiple times (it can be in multiple cliques)
        # Wait - no, each I-vertex can only be in one clique if we're creating Type-B cliques
        # Actually, an I-vertex can be in multiple cliques! The constraint is that each edge
        # is in exactly one clique, not that each vertex is in one clique.
        # So an I-vertex w can be in {u1,v1,w}, {u2,v2,w}, etc.
        # But then cross-edge (u1,w) is covered by the first clique, (u2,w) by the second.
        # This is valid!
        
        # So the matching is: assign each C-C edge to at most one I-vertex.
        # An I-vertex can be assigned to multiple C-C edges.
        # This is actually a set cover / assignment problem, not a standard matching.
        
        # Let me reconsider: if I-vertex w is used in cliques {u1,v1,w} and {u2,v2,w},
        # then cross-edges (u1,w), (v1,w), (u2,w), (v2,w) are all covered.
        # C-C edges (u1,v1) and (u2,v2) are covered.
        # This is valid as long as no edge is covered twice.
        
        # So the constraint is: no C-C edge is covered by two different I-vertices,
        # and no cross-edge is covered by two different cliques.
        
        # A cross-edge (u,w) is covered by clique {u,v,w} if u is in that clique.
        # So if w is in {u1,v1,w} and also in {u2,v2,w}, then (u1,w) is covered by first,
        # (u2,w) by second. No conflict unless u1=u2 or v1=v2, which would mean same C-C edge.
        
        # So the only constraint is: each C-C edge is assigned to at most one I-vertex.
        # This is just: for each C-C edge, pick at most one I-vertex to cover it.
        # To minimize total cliques, we want to maximize the number of C-C edges covered this way.
        
        # But we also need to count the cost: covering C-C edge (u,v) with I-vertex w
        # creates 1 clique that covers 1 C-C edge + 2 cross-edges.
        # Alternative: cover (u,v) with a C-only clique, and cover (u,w), (v,w) with 2-cliques.
        # The Type-B approach saves cliques if the C-only clique would be large.
        
        # Simple strategy: greedily cover C-C edges with Type-B cliques.
        # For each C-C edge, if there's an I-vertex adjacent to both, use it.
        # But we need to be careful about cross-edge conflicts.
        
        # Actually, let's just do: for each C-C edge, if possible, cover with Type-B.
        # Track which cross-edges are covered.
        
        used_cc = set()
        covered_cross = set()  # set of (u,w) pairs covered
        cliques = []
        
        # Process C-C edges in order of most constrained first
        # For each C-C edge, count how many I-vertices can cover it
        cc_order = sorted(range(len(cc_list)), key=lambda i: len(cc_to_iverts[i]))
        
        for idx in cc_order:
            u, v = cc_list[idx]
            if (u, v) in used_cc:
                continue
            # Find an I-vertex that can cover this edge
            for w in cc_to_iverts[idx]:
                # Check if cross-edges (u,w) and (v,w) are not already covered
                if _cc_edge(u, w) not in covered_cross and _cc_edge(v, w) not in covered_cross:
                    cliques.append([u, v, w])
                    used_cc.add((u, v))
                    covered_cross.add(_cc_edge(u, w))
                    covered_cross.add(_cc_edge(v, w))
                    break
        
        # Now cover remaining cross-edges with 2-cliques
        for w in range(k, n):
            for u in sorted(N_C[w]):
                if _cc_edge(u, w) not in covered_cross:
                    cliques.append([u, w])
                    covered_cross.add(_cc_edge(u, w))
        
        # Cover remaining C-C edges with C-only cliques
        remaining_cc = all_cc - used_cc
        c_only = _cover_remaining_cc(remaining_cc, k)
        for cl in c_only:
            if len(cl) > 30:
                pass
            cliques.append(cl)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue
        results.append(solve_one(*spec))

    # convert to plain python int lists
    out = []
    for clist in results:
        nl = []
        for c in clist:
            # Ensure int and list
            nl.append([int(x) for x in c])
        out.append(nl)
    return out
