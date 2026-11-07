            for u in range(n):
                for v in adj[u]:
                    for s in adj[u]:
                        if s == v: continue
                        for t in adj[v]:
                            if t == u or t == s: continue
                            if deg[s] == deg[t]: return False
            