<<<<<<< SEARCH
    if best_edges:
        adj = build_adj_bitmask(set(best_edges))
        found, _ = has_q3_fast(adj)
        if found:
            # This shouldn't happen, but just in case, fall back to K3,13
            best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
=======
    if best_edges:
        adj = build_adj_bitmask(set(best_edges))
        found, _ = has_q3_fast(adj)
        if found:
            # This shouldn't happen, but just in case, fall back to K3,13
            # Note: K3,13 is Q3-free because Q3 is bipartite with parts of size 4,4.
            # Any subgraph of K3,13 has at most 3 vertices in one part of the bipartition of the host,
            # or simply, Q3 requires 8 vertices. K3,13 has max degree 13.
            # Actually, simpler: just return a known sparse graph if verification fails.
            best_edges = []
>>>>>>> REPLACE
