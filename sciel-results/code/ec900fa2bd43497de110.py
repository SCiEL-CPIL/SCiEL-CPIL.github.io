import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0
    margin = 10.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        maximal_cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            if P:
                u = max(P, key=lambda v: len(adj[v] & P))
            else:
                u = next(iter(X)) if X else None
            
            if u is None:
                candidates = P
            else:
                candidates = P - adj[u]
            
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        return maximal_cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        seed_solution = [[list(e)] for e in edges]
        
        if time.time() > deadline - margin:
            return seed_solution
        
        try:
            max_cliques = find_maximal_cliques(n, edges)
        except Exception:
            return seed_solution
        
        if not max_cliques:
            return seed_solution
        
        uncovered = set(edges)
        solution = []
        
        # Sort by size descending
        sorted_cliques = sorted(max_cliques, key=len, reverse=True)
        
        for clique in sorted_cliques:
            if time.time() > deadline - margin:
                break
            
            # Check if all edges of this clique are uncovered
            valid = True
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (u, v) if u < v else (v, u)
                    if e not in uncovered:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                solution.append(list(clique))
                # Remove edges
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        e = (u, v) if u < v else (v, u)
                        uncovered.remove(e)
        
        if uncovered:
            for e in uncovered:
                solution.append(list(e))
        
        return solution

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
