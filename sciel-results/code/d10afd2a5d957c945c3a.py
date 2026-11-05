"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    import time
    deadline = time.time() + 130.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, adj):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            U = P | X
            if not U:
                return
            # Pick pivot with max degree in P
            pivot = max(U, key=lambda u: len(P & adj[u]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R_new, P_new, X_new)
                P = P - {v}
                X = X | {v}
        
        bk(frozenset(), set(range(n)), set())
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        uncovered = list(range(len(cliques)))
        # Map each vertex to the set of clique indices it hits
        vertex_cliques = defaultdict(set)
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)
        
        selected = set()
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in vertex_cliques:
                count = len(vertex_cliques[v] & set(uncovered))
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            selected.add(best_v)
            # Remove all cliques hit by best_v
            uncovered = [i for i in uncovered if i not in vertex_cliques[best_v]]
        
        return selected

    def local_search_improve(selected, cliques, n, max_iter=500):
        """Try to improve by removing vertices that are redundant, or swapping."""
        selected = set(selected)
        
        def is_valid(sel):
            for clique in cliques:
                if not sel & clique:
                    return False
            return True
        
        improved = True
        while improved:
            improved = False
            # Try removing each vertex
            for v in list(selected):
                new_sel = selected - {v}
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for v_out in list(selected):
                for v_in in range(n):
                    if v_in in selected:
                        continue
                    new_sel = (selected - {v_out}) | {v_in}
                    if len(new_sel) < len(selected) and is_valid(new_sel):
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(n, adj)
        
        if not cliques:
            return []
        
        # Greedy hitting set
        selected = greedy_hitting_set(cliques, n)
        
        # Local search improvement
        selected = local_search_improve(selected, cliques, n)
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices as a safe (but large) solution
            results.append(list(range(n)))
            continue
        results.append(solve_instance(n, p, seed))
    
    return results
# EVOLVE-BLOCK-END
