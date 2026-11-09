def _has_path_5(adj, u, v, current, visited_mask, steps):
    if steps == 5:
        return current == v
    # Pruning: if steps remaining > distance to v? 
    # We don't have distance precomputed for G without edge.
    for w in range(N):
        if (adj[current] >> w) & 1:
            if w == u and steps == 4: # Don't return to u too early? No, simple path.
                continue
            if (visited_mask >> w) & 1:
                continue
            if _has_path_5(adj, u, v, w, visited_mask | (1 << w), steps + 1):
                return True
    return False
