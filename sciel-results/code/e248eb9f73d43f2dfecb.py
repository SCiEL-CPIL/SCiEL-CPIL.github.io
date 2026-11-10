    def solve_instance(n, k, p, seed):
        rng = random.Random(seed)
        # Generate edges as in seed, but keep them separate for logic
        internal_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                internal_edges.append((u, v))
                
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
        
        cliques = []
        covered_internal = set()
        
        for v in range(k, n):
            if N[v]:
                clique = sorted(N[v] + [v])
                cliques.append(clique)
                nv = N[v]
                for i in range(len(nv)):
                    for j in range(i + 1, len(nv)):
                        a, b = nv[i], nv[j]
                        if a > b: a, b = b, a
                        covered_internal.add((a, b))
        
        for u, w in internal_edges:
            if (u, w) not in covered_internal:
                cliques.append([u, w])
                
        return cliques
