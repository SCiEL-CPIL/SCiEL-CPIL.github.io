import time
from random import Random

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave 20s margin for 120s limit

    def generate_graph(n, p, seed):
        rng = Random(seed)
        adj = [[False] * n for _ in range(n)]
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.add((u, v))
        return adj, edges

    def bron_kerbosch(adj, n, time_limit, start_time):
        # Returns list of maximal cliques
        cliques = []
        
        def bk(R, P, X):
            # Time check
            if time.time() - start_time > time_limit:
                return
            
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Pivot selection
            P_list = list(P)
            X_list = list(X)
            union = P_list + X_list
            if not union:
                return
            
            best_u = None
            max_inter = -1
            for u in union:
                inter = 0
                for v in P_list:
                    if adj[u][v]:
                        inter += 1
                if inter > max_inter:
                    max_inter = inter
                    best_u = u
            
            if best_u is None:
                return
                
            candidates = [v for v in P if not adj[best_u][v]]
            
            for v in candidates:
                R.add(v)
                # P_new = P ∩ N(v)
                P_new = set()
                for w in P:
                    if adj[v][w]:
                        P_new.add(w)
                # X_new = X ∩ N(v)
                X_new = set()
                for w in X:
                    if adj[v][w]:
                        X_new.add(w)
                
                bk(R, P_new, X_new)
                
                P.remove(v)
                X.add(v)
                R.remove(v)
                
        R = set()
        P = set(range(n))
        X = set()
        bk(R, P, X)
        return cliques

    def get_clique_edges(clique):
        k = len(clique)
        edges = set()
        for i in range(k):
            for j in range(i+1, k):
                u, v = clique[i], clique[j]
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        adj, edges = generate_graph(n, p, seed)
        
        if time.time() - start_time > time_limit:
            return [[u, v] for u, v in edges]

        # Pass 1: Original Graph
        max_cliques = bron_kerbosch(adj, n, time_limit, start_time)
        
        if time.time() - start_time > time_limit:
            return [[u, v] for u, v in edges]

        # Sort by size descending
        max_cliques.sort(key=lambda c: len(c), reverse=True)
        
        uncovered = set(edges)
        partition = []
        
        for clique in max_cliques:
            if time.time() - start_time > time_limit:
                break
                
            clique_edges = get_clique_edges(clique)
            if clique_edges.issubset(uncovered):
                partition.append(clique)
                uncovered -= clique_edges
        
        # Pass 2: Residual Graph
        if uncovered and time.time() - start_time < time_limit - 10: # Keep 10s buffer
            # Build residual adj matrix
            res_adj = [[False] * n for _ in range(n)]
            for u, v in uncovered:
                res_adj[u][v] = True
                res_adj[v][u] = True
            
            res_cliques = bron_kerbosch(res_adj, n, time_limit, start_time)
            
            if time.time() - start_time > time_limit:
                # Fallback for residuals
                for u, v in uncovered:
                    partition.append([u, v])
                return partition

            res_cliques.sort(key=lambda c: len(c), reverse=True)
            
            for clique in res_cliques:
                if time.time() - start_time > time_limit:
                    break
                clique_edges = get_clique_edges(clique)
                if clique_edges.issubset(uncovered):
                    partition.append(clique)
                    uncovered -= clique_edges
        
        # Final Residuals: Edges
        for u, v in uncovered:
            partition.append([u, v])
            
        return partition

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
