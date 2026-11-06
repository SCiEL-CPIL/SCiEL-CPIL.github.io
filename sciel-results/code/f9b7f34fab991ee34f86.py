        for _ in range(max_iter):
            # Find ONE violation
            violator = None
            for u in range(n):
                for v in range(u+1, n):
                    for w in range(v+1, n):
                        common = []
                        for x in range(n):
                            if x != u and x != v and x != w:
                                if adj[u, x] == 1 and adj[v, x] == 1 and adj[w, x] == 1:
                                    common.append(x)
                                    if len(common) >= 3:
                                        break
                        if len(common) >= 3:
                            violator = (u, v, w, common)
                            break
                    if violator: break
                if violator: break
            
            if not violator:
                break # No more violations, graph is valid
            
            u, v, w, common = violator
            x = random.choice(common)
            adj[u, x] = 0
            adj[x, u] = 0
