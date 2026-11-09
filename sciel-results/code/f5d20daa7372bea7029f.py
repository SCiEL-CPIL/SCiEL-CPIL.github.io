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

def _build_disjointness_graph(cliques):
    """
    Build the auxiliary conflict graph where vertices are indices of cliques.
    An edge exists between i and j if cliques[i] and cliques[j] are disjoint.
    Returns adjacency list for this graph.
    """
    m = len(cliques)
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if not (cliques[i] & cliques[j]):
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _chvatal_greedy_max_independent_set(n, adj_list):
    """
    Chvátal's greedy algorithm for maximum independent set (approximation).
    Returns a set of vertex indices (in the auxiliary graph) that form an independent set.
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
        
        # 1. Build auxiliary disjointness graph
        m = len(cliques_to_hit)
        aux_adj = _build_disjointness_graph(cliques_to_hit)
        
        # 2. Find Max Independent Set in the auxiliary graph using Chvatal Greedy
        # The MIS corresponds to a set of pairwise intersecting cliques
        mis_indices = _chvatal_greedy_max_independent_set(m, aux_adj)
        
        # 3. Form initial hitting set as the union of vertices in these cliques
        initial_hitting_set = set()
        for idx in mis_indices:
            initial_hitting_set.update(cliques_to_hit[idx])
        
        # 4. Check for uncovered cliques and repair using greedy set cover
        # A clique is uncovered if it has no intersection with initial_hitting_set
        uncovered_indices = set()
        for i, clique in enumerate(cliques_to_hit):
            if not (clique & initial_hitting_set):
                uncovered_indices.add(i)
        
        # Greedy repair: pick vertices that cover the most uncovered cliques
        while uncovered_indices:
            # Count how many uncovered cliques each candidate vertex covers
            # Candidate vertices are those in the union of uncovered cliques
            candidate_vertices = set()
            for idx in uncovered_indices:
                candidate_vertices.update(cliques_to_hit[idx])
            
            best_vertex = None
            best_count = -1
            
            for v in candidate_vertices:
                count = 0
                for idx in uncovered_indices:
                    if v in cliques_to_hit[idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_vertex = v
            
            if best_vertex is None or best_count == 0:
                # Should not happen if uncovered_indices is not empty
                break
                
            initial_hitting_set.add(best_vertex)
            
            # Remove covered cliques from uncovered set
            new_uncovered = set()
            for idx in uncovered_indices:
                if best_vertex not in cliques_to_hit[idx]:
                    new_uncovered.add(idx)
            uncovered_indices = new_uncovered
            
        results.append(sorted(initial_hitting_set))
    
    return results
# EVOLVE-BLOCK-END
