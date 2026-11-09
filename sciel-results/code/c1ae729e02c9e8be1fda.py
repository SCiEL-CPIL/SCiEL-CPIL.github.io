"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        """
        Find minimum clique partition of a split graph.
        
        In a split graph with clique C (size k) and independent set I (size n-k):
        - Edges within C must be covered by cliques
        - Edges between C and I must be covered by cliques containing exactly one vertex from I
          (since I is independent, no clique can contain two vertices from I)
        - So each vertex v in I that has neighbors in C forms cliques with subsets of its neighbors
        
        Key insight: Each edge (u,v) where v in I must be in a clique containing v and some subset of C.
        Since v is the only vertex from I in any clique, cliques containing different vertices of I are disjoint
        in terms of I-vertices but can share C-vertices.
        
        Actually, the constraint is: every edge must be in EXACTLY one listed clique.
        And every listed clique must be a clique with at least 2 vertices.
        
        For split graphs:
        - The clique C itself can be one clique (covers all edges within C)
        - For each vertex v in I with degree d_v > 0, we need to cover edges (u,v) for u in N(v)
        - A clique containing v can include v and any subset of N(v) (since N(v) subset of C, and C is a clique)
        - So we can cover all edges incident to v with a single clique: {v} union N(v)
        
        But wait - if we use {v} union N(v) for each v in I, then edges within C are covered multiple times
        if they appear in multiple such cliques. We need each edge in EXACTLY one clique.
        
        So the strategy is:
        1. Decide which edges within C to cover by the "C clique" vs by cliques extending to I
        2. For each v in I, choose a subset S_v of N(v) such that {v} union S_v is a clique
           (always true since S_v subset C)
        3. The edges covered by v's clique are: edges within S_v plus edges (u,v) for u in S_v
        4. Edges within C not covered by any S_v must be covered by a clique on C (or subsets)
        
        Actually, let me reconsider. The minimum number of cliques:
        - One clique for C (covers all intra-C edges)
        - For each v in I with N(v) non-empty, one clique {v} union N(v)
        
        But this covers intra-C edges multiple times! Edge (u,w) in C is in the C-clique AND in any
        clique {v} union N(v) where both u,w in N(v).
        
        So we need a partition: each edge in exactly one clique.
        
        Better approach:
        - For each v in I, we must cover edges (u,v) for u in N(v). These edges can only be in cliques
          containing v (since v is in I and I is independent, the only clique containing v has v plus
          some subset of C).
        - So for each v, we pick a clique K_v containing v. K_v = {v} union S_v where S_v subset N(v).
        - The edges (u,v) for u in N(v) are covered by K_v if u in S_v. So we need S_v = N(v) to cover all.
          Wait, no - we need to cover ALL edges (u,v) for u in N(v). So S_v must include all of N(v).
          Therefore K_v = {v} union N(v) is forced (if N(v) is non-empty).
        
        Hmm, but then intra-C edges are over-covered. Let me re-read the problem.
        
        "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        
        So each edge is in exactly one clique. If edge (u,w) with u,w in C is in clique K_v = {v} union N(v),
        then it cannot also be in the C-clique.
        
        So the structure is:
        - Some intra-C edges are covered by "C-only" cliques (subsets of C)
        - Some intra-C edges are covered by "I-extended" cliques {v} union S_v
        - Cross edges (u,v) with v in I are covered by the I-extended clique containing v
        
        For each v in I with N(v) non-empty, the clique containing all cross edges (u,v) must be {v} union N(v)
        (since we need one clique per v to cover all its cross edges, and that clique also covers intra-C edges
        within N(v)).
        
        Wait, can we split v's cross edges among multiple cliques? Yes! We could have:
        - K_v1 = {v} union S_v1
        - K_v2 = {v} union S_v2
        where S_v1 union S_v2 = N(v) and S_v1, S_v2 are subsets.
        But this uses more cliques. To minimize cliques, we want one clique per v (if N(v) non-empty).
        
        So the forced structure:
        - For each v in I with N(v) non-empty: one clique K_v = {v} union N(v)
        - The remaining edges (intra-C edges not covered by any K_v) must be covered by additional cliques on subsets of C.
        
        Which intra-C edges are covered by K_v? Edges (u,w) where u,w in N(v).
        
        So the problem reduces to: cover the graph on C where edge (u,w) is "uncovered" if there's no v in I
        with both u,w in N(v). Cover these remaining edges with minimum number of cliques (which are just
        cliques in the remaining graph, but since C is a clique, any subset is a clique).
        
        The remaining graph on C has edge (u,w) present iff NOT (exists v in I: u in N(v) AND w in N(v)).
        Equivalently, edge (u,w) is present iff the neighborhoods of u and w in I are disjoint.
        
        We need to cover this remaining graph with minimum cliques. Since it's a subgraph of a clique,
        the minimum clique cover equals the chromatic number of the complement of this remaining graph.
        
        Let me compute this for the given instances.
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # C = {0, ..., k-1}, I = {k, ..., n-1}
        # For each v in I, N(v) = {u in C : adj[u][v]}
        N_of_I = []
        for v in range(k, n):
            neighbors = [u for u in range(k) if adj[u][v]]
            N_of_I.append(neighbors)
        
        # Determine which intra-C edges are covered by I-extended cliques
        # Edge (u,w) in C is covered if exists v in I with u in N(v) and w in N(v)
        covered = [[False] * k for _ in range(k)]
        for neighbors in N_of_I:
            for i, u in enumerate(neighbors):
                for w in neighbors[i+1:]:
                    covered[u][w] = True
                    covered[w][u] = True
        
        # Remaining graph on C: edge (u,w) present if not covered[u][w]
        # Need to cover this with minimum cliques
        # Since it's a subgraph of a clique, min clique cover = chromatic number of complement
        
        # Build complement of remaining graph (within C)
        # Complement edge (u,w) exists if covered[u][w] is True (i.e., u,w have common neighbor in I)
        # Wait, let me think again.
        
        # Remaining graph G_rem on vertices C: edge (u,w) iff not covered[u][w]
        # We want to cover G_rem with minimum cliques.
        # A clique in G_rem is a set of vertices where all pairs are edges in G_rem.
        # Min clique cover of G_rem = min number of cliques whose union of edges covers all edges of G_rem.
        
        # Since G_rem is a subgraph of a complete graph, a clique in G_rem is just a subset of vertices
        # where all pairwise edges exist in G_rem.
        
        # This is the same as: partition the edges of G_rem into minimum cliques.
        # Equivalently, color the vertices of the complement of G_rem (within C) such that...
        
        # Actually, min clique edge cover of a graph G equals the chromatic number of the line graph...
        # No, let me just use a greedy approach or exact for small k.
        
        # For k up to 30, we can try to find a good solution.
        # The remaining graph G_rem: let's build it.
        
        # Vertices 0..k-1, edge (u,w) if not covered[u][w]
        # Find min clique edge cover.
        
        # Greedy: repeatedly pick the largest clique in remaining edges, remove those edges.
        # Or: since k is small, try a few strategies.
        
        # Build list of remaining edges
        rem_edges = []
        for u in range(k):
            for w in range(u+1, k):
                if not covered[u][w]:
                    rem_edges.append((u, w))
        
        # If no remaining edges, we just need the I-extended cliques
        if not rem_edges:
            cliques = []
            for v in range(k, n):
                neighbors = N_of_I[v - k]
                if len(neighbors) > 0:
                    clique = [v] + neighbors
                    if len(clique) >= 2:
                        cliques.append(clique)
            return cliques
        
        # Need to cover rem_edges with cliques (subsets of C where all pairs are in rem_edges)
        # Plus the I-extended cliques
        
        # Greedy approach: find a set of cliques covering all rem_edges
        # Since the graph is a subgraph of K_k, we can use a simple strategy:
        # The complement of G_rem (call it H) has edge (u,w) iff covered[u][w].
        # A clique in G_rem is an independent set in H.
        # Min clique edge cover of G_rem... 
        
        # Simple greedy: repeatedly find a large clique in remaining G_rem, use it.
        import copy
        
        remaining = set(rem_edges)
        cliques_on_C = []
        
        # Build adjacency for remaining graph
        rem_adj = [[False] * k for _ in range(k)]
        for u, w in rem_edges:
            rem_adj[u][w] = True
            rem_adj[w][u] = True
        
        while remaining:
            # Find a large clique in the remaining graph
            # Greedy: pick vertex with highest degree, then find max clique containing it
            # Simple: try all pairs/triples or use greedy
            best_clique = None
            best_size = 0
            
            # Try single vertices (clique of size 1 covers no edges, so skip)
            # Try all pairs
            for u, w in list(remaining):
                # Check if (u,w) is still in remaining
                if (u, w) not in remaining and (w, u) not in remaining:
                    continue
                
                # Try to extend to a clique
                # Common neighbors of u and w in remaining graph
                common = [x for x in range(k) if rem_adj[u][x] and rem_adj[w][x] and x != u and x != w]
                
                # Greedy extend
                clique = [u, w]
                candidates = common[:]
                while candidates:
                    # Pick candidate with most connections to current clique
                    best_c = None
                    best_score = -1
                    for c in candidates:
                        score = sum(1 for v in clique if rem_adj[c][v])
                        if score > best_score:
                            best_score = score
                            best_c = c
                    if best_c is not None and best_score == len(clique):
                        clique.append(best_c)
                        candidates.remove(best_c)
                        # Update candidates
                        candidates = [c for c in candidates if all(rem_adj[c][v] for v in clique)]
                    else:
                        break
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or len(best_clique) < 2:
                # Shouldn't happen if remaining is non-empty
                break
            
            # Remove edges covered by best_clique
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    u, w = best_clique[i], best_clique[j]
                    if (u, w) in remaining:
                        remaining.discard((u, w))
                    if (w, u) in remaining:
                        remaining.discard((w, u))
            
            cliques_on_C.append(best_clique)
        
        # Build final clique list
        result = []
        
        # Add I-extended cliques
        for v in range(k, n):
            neighbors = N_of_I[v - k]
            if len(neighbors) > 0:
                clique = [v] + neighbors
                if len(clique) >= 2:
                    result.append(clique)
        
        # Add cliques on C
        for clique in cliques_on_C:
            if len(clique) >= 2:
                result.append(clique)
        
        return result

    # Solve all three instances
    specs = [(30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)]
    results = []
    
    for spec in specs:
        if time.time() > DEADLINE:
            # Fallback to trivial solution
            break
        cliques = solve_split_graph(*spec)
        results.append(cliques)
    
    # If we ran out of time, use trivial fallback for any missing
    while len(results) < 3:
        idx = len(results)
        n, k, p, seed = specs[idx]
        edges = instance(n, k, p, seed)
        results.append([list(e) for e in edges])
    
    return results
# EVOLVE-BLOCK-END
