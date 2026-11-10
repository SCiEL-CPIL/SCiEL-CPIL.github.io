def is_valid_graph(n, adj, deadline):
    for k in range(6, n+1):
        for s in itertools.combinations(range(n), k):
            if time.time() > deadline:
                return None # Timeout
            degs = [0]*k
            for i in range(k):
                for j in range(k):
                    if i != j:
                        degs[i] += adj[s[i]][s[j]]
            if all(d == degs[0] for d in degs):
                return False
    return True
