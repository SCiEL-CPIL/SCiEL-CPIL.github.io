<<<<<<< SEARCH
def _maximal_sets(n, adj, is_clique):
    """Return all maximal cliques (if is_clique) or maximal independent sets."""
    # Bron-Kerbosch with pivoting
    results = []

    def bronk(R, P_set, X):
        if not P_set and not X:
            results.append(frozenset(R))
            return
        # Choose pivot u from P_set ∪ X
        union = P_set | X
        if not union:
            return
        # Pick pivot with max degree in P_set
        best_u = max(union, key=lambda u: len(P_set & adj[u]))
        candidates = P_set - adj[best_u] if is_clique else P_set - (set(range(n)) - adj[best_u] - {best_u})
        # For independent sets, candidates are those NOT adjacent to best_u (excluding best_u itself)
        if not is_clique:
            candidates = P_set - adj[best_u]
            # Also exclude best_u if it's in P_set
            candidates.discard(best_u)
        else:
            candidates = P_set - adj[best_u]

        for v in list(candidates):
            bronk(R | {v}, P_set & adj[v] if is_clique else P_set & (set(range(n)) - adj[v] - {v}),
                  X & adj[v] if is_clique else X & (set(range(n)) - adj[v] - {v}))
            P_set.remove(v)
            X.add(v)

    all_vertices = frozenset(range(n))
    bronk(frozenset(), all_vertices, frozenset())
    return [list(s) for s in results]
=======
def _maximal_sets(n, adj, is_clique):
    """Return all maximal cliques (if is_clique) or maximal independent sets."""
    # Bron-Kerbosch with pivoting
    results = []
    all_vertices = set(range(n))

    def bronk(R, P_set, X):
        if not P_set and not X:
            results.append(frozenset(R))
            return
        # Choose pivot u from P_set ∪ X
        union = P_set | X
        if not union:
            return
        # Pick pivot with max degree in P_set
        if is_clique:
            best_u = max(union, key=lambda u: len(P_set & adj[u]))
            candidates = P_set - adj[best_u]
        else:
            # For independent sets, neighbors in the complement graph are non-neighbors in original graph
            # P_set contains vertices that can be added.
            # We want vertices in P_set that are NOT adjacent to best_u.
            # adj[best_u] contains neighbors in original graph.
            # So non-neighbors are all_vertices - adj[best_u] - {best_u}.
            # But we only care about P_set.
            best_u = max(union, key=lambda u: len(P_set & (all_vertices - adj[u] - {u})))
            candidates = P_set & (all_vertices - adj[best_u] - {best_u})

        for v in list(candidates):
            if is_clique:
                new_P = P_set & adj[v]
                new_X = X & adj[v]
            else:
                new_P = P_set & (all_vertices - adj[v] - {v})
                new_X = X & (all_vertices - adj[v] - {v})
            
            bronk(R | {v}, new_P, new_X)
            P_set.remove(v)
            X.add(v)

    bronk(frozenset(), all_vertices, frozenset())
    return [list(s) for s in results]
>>>>>>> REPLACE

