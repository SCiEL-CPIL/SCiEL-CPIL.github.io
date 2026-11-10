"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
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
            
            # Pivot selection: choose vertex with max |P ∩ N(v)|
            pivot = max(P | X, key=lambda v: len(P & adj[v])) if (P | X) else None
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        expand(frozenset(), set(range(n)), set())
        return maximal_cliques
    
    def greedy_set_cover(cliques, n, time_limit=5.0):
        """Greedy set cover with local search improvement."""
        start = time.time()
        
        # Map each clique to its index
        clique_list = list(cliques)
        num_cliques = len(clique_list)
        
        if num_cliques == 0:
            return set()
        
        # For each vertex, which cliques does it cover?
        vertex_cliques = {v: set() for v in range(n)}
        for i, clique in enumerate(clique_list):
            for v in clique:
                vertex_cliques[v].add(i)
        
        # Greedy initialization
        covered = set()
        selected = set()
        
        # Precompute for efficiency
        uncovered_count = {v: len(vertex_cliques[v]) for v in range(n)}
        
        while len(covered) < num_cliques:
            if time.time() - start > time_limit:
                break
            
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                cnt = len(vertex_cliques[v] - covered)
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            
            if best_v is None or best_count <= 0:
                break
            
            selected.add(best_v)
            covered.update(vertex_cliques[best_v])
        
        # Local search: try to improve
        for _ in range(50):
            if time.time() - start > time_limit:
                break
            
            improved = False
            # Try removing a vertex
            for v in list(selected):
                new_selected = selected - {v}
                new_covered = set()
                for u in new_selected:
                    new_covered.update(vertex_cliques[u])
                if new_covered == set(range(num_cliques)):
                    selected = new_selected
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add one
            for v_remove in list(selected):
                for v_add in range(n):
                    if v_add in selected:
                        continue
                    new_selected = selected - {v_remove} | {v_add}
                    new_covered = set()
                    for u in new_selected:
                        new_covered.update(vertex_cliques[u])
                    if new_covered == set(range(num_cliques)) and len(new_selected) < len(selected):
                        selected = new_selected
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline:
            # Emergency fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        # Filter to only cliques with >= 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy set cover
        selected = greedy_set_cover(cliques, n, time_limit=min(10.0, deadline - time.time() - 2))
        
        # Verify coverage
        covered_cliques = set()
        for clique in cliques:
            for v in selected:
                if v in clique:
                    covered_cliques.add(clique)
                    break
        
        if len(covered_cliques) != len(cliques):
            # Fallback: add vertices to cover remaining
            for clique in cliques:
                if clique not in covered_cliques:
                    for v in clique:
                        if v not in selected:
                            selected.add(v)
                            break
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
