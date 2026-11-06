def check_c5_in_5(verts, adj):
    u, v, w, x, y = verts
    # 12 cycles
    if (adj[u]>>v)&1 and (adj[v]>>w)&1 and (adj[w]>>x)&1 and (adj[x]>>y)&1 and (adj[y]>>u)&1: return True
    if (adj[u]>>v)&1 and (adj[v]>>w)&1 and (adj[w]>>y)&1 and (adj[y]>>x)&1 and (adj[x]>>u)&1: return True
    if (adj[u]>>v)&1 and (adj[v]>>x)&1 and (adj[x]>>w)&1 and (adj[w]>>y)&1 and (adj[y]>>u)&1: return True
    if (adj[u]>>v)&1 and (adj[v]>>x)&1 and (adj[x]>>y)&1 and (adj[y]>>w)&1 and (adj[w]>>u)&1: return True
    if (adj[u]>>v)&1 and (adj[v]>>y)&1 and (adj[y]>>w)&1 and (adj[w]>>x)&1 and (adj[x]>>u)&1: return True
    if (adj[u]>>v)&1 and (adj[v]>>y)&1 and (adj[y]>>x)&1 and (adj[x]>>w)&1 and (adj[w]>>u)&1: return True
    if (adj[u]>>w)&1 and (adj[w]>>v)&1 and (adj[v]>>x)&1 and (adj[x]>>y)&1 and (adj[y]>>u)&1: return True
    if (adj[u]>>w)&1 and (adj[w]>>v)&1 and (adj[v]>>y)&1 and (adj[y]>>x)&1 and (adj[x]>>u)&1: return True
    if (adj[u]>>w)&1 and (adj[w]>>x)&1 and (adj[x]>>v)&1 and (adj[v]>>y)&1 and (adj[y]>>u)&1: return True
    if (adj[u]>>w)&1 and (adj[w]>>y)&1 and (adj[y]>>v)&1 and (adj[v]>>x)&1 and (adj[x]>>u)&1: return True
    if (adj[u]>>x)&1 and (adj[x]>>v)&1 and (adj[v]>>w)&1 and (adj[w]>>y)&1 and (adj[y]>>u)&1: return True
    if (adj[u]>>x)&1 and (adj[x]>>y)&1 and (adj[y]>>v)&1 and (adj[v]>>w)&1 and (adj[w]>>u)&1: return True
    return False
