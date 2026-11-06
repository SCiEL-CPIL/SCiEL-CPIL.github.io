def construct():
    import sys
    sys.setrecursionlimit(10000)
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    import random

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                cliques.append(frozenset(R))
                return
            pivot_set = P | X
            if pivot_set:
                pivot = max(pivot_set, key=lambda v: sum(1 for u in P if adj[v][u]))
                candidates = [v for v in P if not adj[v][pivot]]
            else:
                candidates = list(P)
            
            for v in candidates:
                bk(R | {v}, P & {u for u in P if adj[v][u]}, X & {u for u in X if adj[v][u]})
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_hitting_set_ilp(cliques, n):
        if not cliques:
            return set()
        
        c = np.ones(n)
        A = np.zeros((len(cliques), n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        lin_cons = LinearConstraint(A, lb=1, ub=np.inf)
        
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds)
        
        if res.status != 0:
            # Fallback: simple greedy
            return greedy_fallback(cliques, n)
            
        return {i for i in range(n) if res.x[i] > 0.5}

    def greedy_fallback(cliques, n):
        if not cliques:
            return set()
        unhit = list(cliques)
        result = set()
        while unhit:
            counts = {}
            for c in unhit:
                for v in c:
                    counts[v] = counts.get(v, 0) + 1
            if not counts:
                break
            best_v = max(counts, key=lambda v: counts[v])
            result.add(best_v)
            unhit = [c for c in unhit if best_v not in c]
        return result

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = generate_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]
        
        try:
            hitting_set = solve_hitting_set_ilp(relevant_cliques, n)
        except Exception:
            hitting_set = greedy_fallback(relevant_cliques, n)
            
        out.append(sorted(hitting_set))
    
    return out
