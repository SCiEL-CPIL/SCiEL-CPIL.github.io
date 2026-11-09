                                    for u in range(n):
                                        for v in range(u+1, n):
                                            if adj[u] & (1 << v): # u,v adjacent
                                                S = adj[u] & adj[v]
                                                if has_k22_in_set(S, adj):
                                                    return True
                                    return False
                                    