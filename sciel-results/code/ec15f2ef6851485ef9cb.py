"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
import sys

def _random_graph_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_maximal_cliques(n, adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot selection.
    Returns list of frozensets of vertices.
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        P_union_X = P | X
        if P_union_X:
            # Pick pivot with max |P ∩ N(u)|
            best_u = max(P_union_X, key=lambda u: len(P & adj[u]))
        else:
            return
        
        # Iterate over P \ N(u)
        for v in list(P - adj[best_u]):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _chvatal_greedy_max_independent_set(n, adj_list):
    """
    Chvátal's greedy algorithm for maximum independent set (approximation).
    Returns a set of vertices that form an independent set.
    """
    remaining = set(range(n))
    independent = set()
    
    while remaining:
        # Pick vertex with minimum degree in the induced subgraph on remaining
        min_deg = float('inf')
        chosen = None
        for v in remaining:
            deg = len(adj_list[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                chosen = v
        
        independent.add(chosen)
        remaining.remove(chosen)
        # Remove all neighbors of chosen from remaining
        remaining -= adj_list[chosen]
    
    return independent

def _solve_hitting_set(maximal_cliques, n):
    """
    Solves the hitting set problem using the auxiliary conflict graph approach.
    1. Build graph H where vertices are maximal cliques (size >= 2).
    2. Edge between cliques if they are disjoint.
    3. Find Max Independent Set (MIS) in H using Chvatal's greedy.
    4. Union of cliques in MIS forms a core hitting set candidate.
    5. Augment with greedy set cover for any unhit cliques.
    6. Prune redundant vertices.
    """
    if not maximal_cliques:
        return set()
    
    # Filter to cliques of size >= 2
    cliques = [c for c in maximal_cliques if len(c) >= 2]
    
    if not cliques:
        return set()
    
    m = len(cliques)
    
    # Build auxiliary graph adjacency
    # adj_aux[i] is a set of j such that cliques[i] and cliques[j] are disjoint
    adj_aux = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if not (cliques[i] & cliques[j]):
                adj_aux[i].add(j)
                adj_aux[j].add(i)
    
    # Find MIS in auxiliary graph using existing Chvatal helper
    mis_indices = _chvatal_greedy_max_independent_set(m, adj_aux)
    
    # Form core hitting set
    core_set = set()
    for idx in mis_indices:
        core_set.update(cliques[idx])
    
    # Check which cliques are not hit by core_set
    unhit_cliques = []
    for c in cliques:
        if not (c & core_set):
            unhit_cliques.append(c)
    
    # Augment with greedy set cover for unhit cliques
    if unhit_cliques:
        uncovered = set(range(len(unhit_cliques)))
        while uncovered:
            best_vertex = None
            best_count = -1
            for v in range(n):
                count = 0
                for idx in uncovered:
                    if v in unhit_cliques[idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_vertex = v
            
            if best_vertex is None or best_count == 0:
                break
                
            core_set.add(best_vertex)
            new_uncovered = set()
            for idx in uncovered:
                if best_vertex not in unhit_cliques[idx]:
                    new_uncovered.add(idx)
            uncovered = new_uncovered
            
    # Prune redundant vertices
    final_set = set(core_set)
    for v in list(final_set):
        candidate_set = final_set - {v}
        is_valid = True
        for c in cliques:
            if not (c & candidate_set):
                is_valid = False
                break
        if is_valid:
            final_set.remove(v)
            
    return final_set

def construct():
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        maximal_cliques = _bron_kerbosch_maximal_cliques(n, adj)
        
        # Filter to cliques of size >= 2
        cliques_to_hit = [c for c in maximal_cliques if len(c) >= 2]
        
        if not cliques_to_hit:
            results.append([])
            continue
        
        # Use greedy hitting set
        hitting_set = _solve_hitting_set(cliques_to_hit, n)
        results.append(sorted(hitting_set))
    
    return results
# EVOLVE-BLOCK-END
