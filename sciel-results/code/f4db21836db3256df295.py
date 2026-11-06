"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    
    def bron_kerbosch(R, P, X, adj, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        best_pivot = None
        best_count = -1
        for v in P:
            count = len(P & adj[v])
            if count > best_count:
                best_count = count
                best_pivot = v
        for v in X:
            count = len(P & adj[v])
            if count > best_count:
                best_count = count
                best_pivot = v
        
        if best_pivot is None:
            candidates = list(P)
        else:
            candidates = [v for v in P if v not in adj[best_pivot]]
        
        for v in candidates:
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques)
            P = P - {v}
            X = X | {v}
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def greedy_transversal(cliques, n):
        if not cliques:
            return []
        
        vertex_clique_count = [0] * n
        for clique in cliques:
            for v in clique:
                vertex_clique_count[v] += 1
        
        uncovered = set(range(len(cliques)))
        selected = []
        
        while uncovered:
            best_v = None
            best_count = 0
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for i in uncovered if v in cliques[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
            
            selected.append(best_v)
            new_uncovered = set()
            for i in uncovered:
                if best_v not in cliques[i]:
                    new_uncovered.add(i)
            uncovered = new_uncovered
        
        return selected
    
    def local_search_improve(selected, cliques, n, max_iterations=50):
        selected_set = set(selected)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for v in list(selected_set):
                temp_selected = selected_set - {v}
                
                uncovered = []
                for i, clique in enumerate(cliques):
                    if not (clique & temp_selected):
                        uncovered.append(clique)
                
                if not uncovered:
                    selected_set = temp_selected
                    improved = True
                    break
        
        return list(selected_set)
    
    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        
        cliques = []
        bron_kerbosch(frozenset(), frozenset(range(n)), frozenset(), adj, cliques)
        
        if not cliques:
            return []
        
        cliques_list = [set(c) for c in cliques]
        
        selected = greedy_transversal(cliques_list, n)
        
        selected = local_search_improve(selected, cliques_list, n, max_iterations=50)
        
        return sorted(selected)
    
    start_time = time.time()
    deadline = start_time + 140
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue
        
        result = solve_instance(n, p, seed)
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
