    def update_cn(adj, cn, u, v, sign):
        # sign=1 for add, sign=-1 for remove
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:
                cn[u, w] += sign
                cn[w, u] += sign
            if adj[u, w]:
                cn[v, w] += sign
                cn[w, v] += sign
