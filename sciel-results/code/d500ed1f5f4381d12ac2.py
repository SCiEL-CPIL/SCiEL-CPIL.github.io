"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    """Generate a random graph using the same RNG sequence as the seed program."""
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _bron_kerbosch_maximal_cliques(n, adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    adj: list of sets, adj[i] is the set of neighbors of i.
    Returns list of sets (each a clique).
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        U = P | X
        if not U:
            return
            
        # Find pivot u that maximizes |P ∩ N(u)|
        best_u = None
        max_count = -1
        for u in U:
            count = len(P & adj[u])
            if count > max_count:
                max_count = count
                best_u = u
                
        u = best_u
        candidates = P - adj[u]
        
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.remove(v)
            X.add(v)
            
    initial_P = set(range(n))
    bk(set(), initial_P, set())
    return [set(c) for c in cliques]

def _solve_set_cover_greedy_local_search(vertices, cliques, n):
    """
    Solve set cover using Greedy + Local Search.
    """
    if not cliques:
        return set()
    
    # Precompute coverage: vertex -> set of clique indices
    vertex_to_cliques = defaultdict(set)
    for ci, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].add(ci)
            
    # 1. Greedy Initialization
    selected = set()
    uncovered = set(range(len(cliques)))
    
    # Greedy: pick v that covers most uncovered cliques.
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            # Count uncovered cliques covered by v
            count = 0
            for ci in vertex_to_cliques[v]:
                if ci in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v is None:
            # Should not happen if cliques are valid
            break
            
        selected.add(best_v)
        # Remove covered cliques
        for ci in vertex_to_cliques[best_v]:
            if ci in uncovered:
                uncovered.remove(ci)
                
    # 2. Local Search
    # We will try to improve the solution by:
    # a) Removing any vertex that is not essential.
    # b) Replacing a set of k vertices with j < k vertices.
    
    improved = True
    while improved:
        improved = False
        
        # a) Remove non-essential vertices
        for v in list(selected):
            # Check if v is the only cover for any clique
            essential = False
            for ci in vertex_to_cliques[v]:
                # If this clique is covered ONLY by v in selected, v is essential
                covers = [u for u in selected if ci in vertex_to_cliques[u]]
                if len(covers) == 1:
                    essential = True
                    break
            if not essential:
                selected.remove(v)
                improved = True
                # Restart loop as selection changed
                break
                
        if improved:
            continue
            
        # b) Try 2-for-1 swap: Replace {u, v} with {w} if w covers all cliques covered by u and v.
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                u, v = sel_list[i], sel_list[j]
                # Union of cliques covered by u and v
                union_cliques = vertex_to_cliques[u] | vertex_to_cliques[v]
                
                # Find a vertex w not in selected that covers all cliques in union_cliques
                for w in range(n):
                    if w in selected:
                        continue
                    if vertex_to_cliques[w].issuperset(union_cliques):
                        # Valid swap
                        selected.remove(u)
                        selected.remove(v)
                        selected.add(w)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
        # c) Try 3-for-1 swap
        if not improved:
             sel_list = list(selected)
             for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    for k in range(j + 1, len(sel_list)):
                        u, v, w_out = sel_list[i], sel_list[j], sel_list[k]
                        union_cliques = vertex_to_cliques[u] | vertex_to_cliques[v] | vertex_to_cliques[w_out]
                        for w_in in range(n):
                            if w_in in selected:
                                continue
                            if vertex_to_cliques[w_in].issuperset(union_cliques):
                                selected.remove(u)
                                selected.remove(v)
                                selected.remove(w_out)
                                selected.add(w_in)
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break

    return selected

def construct():
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = _random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques of size >= 2
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        cliques = [c for c in cliques if len(c) >= 2]
        
        # Solve set cover
        if not cliques:
            out.append([])
        else:
            selected = _solve_set_cover_greedy_local_search(range(n), cliques, n)
            out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END
