   e = rng.choice(list(edges))
   u, v = e
   # remove e
   adj[u] &= ~(1 << v)
   adj[v] &= ~(1 << u)
   edges.discard(e)
   