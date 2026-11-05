    for (y, _) in VADJ[v]:
        if y == u: continue
        if adj[u] >> y & 1: continue # u,y adjacent
        if (adj[u] & adj[y]).bit_count() == 2: return False
    