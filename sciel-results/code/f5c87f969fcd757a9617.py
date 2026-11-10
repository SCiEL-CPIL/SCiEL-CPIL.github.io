"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import sys
    from itertools import combinations
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            # Pivot selection: choose vertex in P ∪ X with maximum |P ∩ N(v)|
            pivot_pool = P | X
            if not pivot_pool:
                return
            pivot = max(pivot_pool, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques
    
    def greedy_set_cover(cliques, universe):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        uncovered = set(range(len(cliques)))
        # Map each vertex to the set of clique indices it belongs to
        vertex_to_cliques = {v: set() for v in universe}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        selected = set()
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in universe:
                if v in selected:
                    continue
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        return sorted(selected)
    
    def improved_greedy_with_local_search(cliques, universe, adj):
        """Greedy set cover followed by local search to remove redundant vertices."""
        # Start with greedy solution
        uncovered = set(range(len(cliques)))
        vertex_to_cliques = {v: set() for v in universe}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        selected = set()
        while uncovered:
            best_v = None
            best_count = -1
            for v in universe:
                if v in selected:
                    continue
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            uncovered -= vertex_to_cliques[best_v]
        
        # Local search: try to remove each vertex if it's not needed
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                # Check if removing v still covers all cliques
                # A clique is covered if it has at least one vertex in selected
                # After removing v, a clique is uncovered only if v was its only selected vertex
                needed = False
                for i in vertex_to_cliques[v]:
                    clique = cliques[i]
                    # Check if any other vertex in this clique is in selected
                    other_covered = False
                    for u in clique:
                        if u != v and u in selected:
                            other_covered = True
                            break
                    if not other_covered:
                        needed = True
                        break
                if not needed:
                    selected.remove(v)
                    improved = True
        
        return sorted(selected)
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        # Get all vertices that appear in any maximal clique of size >= 2
        universe = set()
        for clique in cliques:
            universe.update(clique)
        
        if not cliques:
            out.append([])
            continue
        
        solution = improved_greedy_with_local_search(cliques, universe, adj)
        out.append(solution)
    
    return out
# EVOLVE-BLOCK-END
