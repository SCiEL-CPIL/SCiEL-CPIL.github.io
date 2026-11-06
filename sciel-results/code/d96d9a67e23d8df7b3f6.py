"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def _generate_graph(n, p, seed):
    """Generate a random graph using the same method as the seed program."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_bitmask_adj(n, edges):
    """Build adjacency as bitmasks for fast operations."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _bron_kerbosch(R, P, X, adj, cliques, min_size=2):
    """Bron-Kerbosch with pivot to enumerate maximal cliques of size >= min_size."""
    if not P and not X:
        if len(R) >= min_size:
            cliques.append(tuple(sorted(R)))
        return
    
    # Choose pivot: vertex in P ∪ X with max |P ∩ N(v)|
    best_pivot = None
    best_count = -1
    # Check P first
    for v in P:
        count = bin(P & adj[v]).count('1')
        if count > best_count:
            best_count = count
            best_pivot = v
    # Check X
    for v in X:
        count = bin(P & adj[v]).count('1')
        if count > best_count:
            best_count = count
            best_pivot = v
    
    if best_pivot is None:
        # No pivot found, iterate over all P
        candidates = list(P)
    else:
        # Only need to iterate over P \ N(pivot)
        candidates = [v for v in P if not (adj[best_pivot] & (1 << v))]
    
    for v in candidates:
        bit = 1 << v
        R_new = R | bit
        P_new = P & adj[v]
        X_new = X & adj[v]
        _bron_kerbosch(R_new, P_new, X_new, adj, cliques, min_size)
        P &= ~bit
        X |= bit

def _find_maximal_cliques(n, edges):
    """Find all maximal cliques of size >= 2 using Bron-Kerbosch."""
    adj = _build_bitmask_adj(n, edges)
    cliques = []
    all_vertices = (1 << n) - 1
    _bron_kerbosch(0, all_vertices, 0, adj, cliques, min_size=2)
    return cliques

def _greedy_transversal(cliques, n, weight_strategy="count"):
    """
    Greedy set cover: pick vertices to cover all cliques.
    weight_strategy: 'count' (most cliques covered), 'size' (smallest cliques first), 
                     'inverse' (inverse of number of cliques a vertex is in)
    """
    if not cliques:
        return set()
    
    # For each vertex, count how many cliques it belongs to
    vertex_clique_count = [0] * n
    for clique in cliques:
        for v in clique:
            vertex_clique_count[v] += 1
    
    # Track which cliques are still uncovered
    uncovered = set(range(len(cliques)))
    selected = set()
    
    # Precompute for each vertex, which cliques it covers
    vertex_covers = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_covers[v].add(i)
    
    while uncovered:
        best_v = None
        best_score = -1
        
        if weight_strategy == "count":
            # Pick vertex covering most uncovered cliques
            for v in range(n):
                if v in selected:
                    continue
                covered = len(vertex_covers[v] & uncovered)
                if covered > best_score:
                    best_score = covered
                    best_v = v
        elif weight_strategy == "size":
            # Prioritize covering smallest cliques first
            min_clique_size = min(len(cliques[i]) for i in uncovered)
            for v in range(n):
                if v in selected:
                    continue
                covered_small = 0
                for i in uncovered:
                    if v in cliques[i] and len(cliques[i]) == min_clique_size:
                        covered_small += 1
                # Tie-break by total coverage
                total_covered = len(vertex_covers[v] & uncovered)
                score = covered_small * 1000 + total_covered
                if score > best_score:
                    best_score = score
                    best_v = v
        elif weight_strategy == "inverse":
            # Pick vertex with lowest count (least "busy") that covers uncovered cliques
            for v in range(n):
                if v in selected:
                    continue
                covered = len(vertex_covers[v] & uncovered)
                if covered == 0:
                    continue
                score = covered / (vertex_clique_count[v] + 1)
                if score > best_score:
                    best_score = score
                    best_v = v
        
        if best_v is None or best_score <= 0:
            # Fallback: pick any vertex in an uncovered clique
            any_clique = next(iter(uncovered))
            best_v = cliques[any_clique][0]
        
        selected.add(best_v)
        uncovered -= vertex_covers[best_v]
    
    return selected

def _local_search_reduce(selected, cliques, n, max_iterations=1000):
    """
    Local search: try to remove vertices from the transversal if possible.
    A vertex can be removed if every clique it covers is also covered by another selected vertex.
    """
    if not selected:
        return selected
    
    selected = set(selected)
    
    # Precompute for each vertex, which cliques it covers
    vertex_covers = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_covers[v].add(i)
    
    # For each clique, which selected vertices cover it
    clique_covered_by = defaultdict(set)
    for v in selected:
        for i in vertex_covers[v]:
            clique_covered_by[i].add(v)
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try removing each vertex
        for v in list(selected):
            # Check if all cliques covered by v are also covered by other selected vertices
            can_remove = True
            for i in vertex_covers[v]:
                others = clique_covered_by[i] - {v}
                if not others:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove v
                selected.discard(v)
                for i in vertex_covers[v]:
                    clique_covered_by[i].discard(v)
                    if not clique_covered_by[i]:
                        # This would leave clique i uncovered - shouldn't happen if can_remove is True
                        pass
                improved = True
    
    return selected

def _local_search_swap(selected, cliques, n, max_iterations=500):
    """
    Local search: try swapping one vertex for another to reduce size or maintain.
    Actually, let's try a different approach: for each vertex in selected, 
    try replacing it with a vertex not in selected that maintains coverage.
    """
    if not selected:
        return selected
    
    selected = set(selected)
    
    # Precompute
    vertex_covers = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_covers[v].add(i)
    
    # For each clique, which vertices in the graph cover it
    clique_vertices = [set(clique) for clique in cliques]
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to remove each vertex and see if we can fix coverage
        for v in list(selected):
            # Temporarily remove v
            selected.discard(v)
            
            # Find uncovered cliques
            uncovered = []
            for i, clique in enumerate(cliques):
                if not (set(clique) & selected):
                    uncovered.append(i)
            
            if not uncovered:
                # v was redundant, keep it removed
                improved = True
                continue
            
            # Try to add a single vertex to cover all uncovered cliques
            best_add = None
            best_score = -1
            for w in range(n):
                if w in selected:
                    continue
                # Check if w covers all uncovered cliques
                covers_all = True
                for i in uncovered:
                    if w not in cliques[i]:
                        covers_all = False
                        break
                if covers_all:
                    # Score: prefer vertices that cover more total cliques
                    score = len(vertex_covers[w] & set(uncovered))
                    if score > best_score:
                        best_score = score
                        best_add = w
            
            if best_add is not None:
                selected.add(best_add)
                improved = True
            else:
                # Can't fix with one vertex, restore v
                selected.add(v)
    
    return selected

def construct():
    """
    Construct vertex transversals for three random graphs.
    Returns three lists, one per instance.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        # Check time
        if time.time() > deadline - 30:
            # Not enough time for this instance, use a quick fallback
            edges = _generate_graph(n, p, seed)
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
            continue
        
        # Generate graph
        edges = _generate_graph(n, p, seed)
        
        # Find all maximal cliques of size >= 2
        cliques = _find_maximal_cliques(n, edges)
        
        # If no cliques, return empty
        if not cliques:
            results.append([])
            continue
        
        # Try multiple strategies and keep the best
        best_solution = None
        best_size = float('inf')
        
        strategies = ["count", "size", "inverse"]
        
        for strategy in strategies:
            if time.time() > deadline - 20:
                break
            
            # Greedy initialization
            sol = _greedy_transversal(cliques, n, weight_strategy=strategy)
            
            # Local search reduction
            sol = _local_search_reduce(sol, cliques, n, max_iterations=500)
            
            # Local search swap
            sol = _local_search_swap(sol, cliques, n, max_iterations=200)
            
            # Verify solution is valid
            valid = True
            for clique in cliques:
                if not (set(clique) & sol):
                    valid = False
                    break
            
            if valid and len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
        
        # If no valid solution found (shouldn't happen), fallback
        if best_solution is None:
            # Fallback: all vertices in any edge
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            best_solution = touched
        
        results.append(sorted(best_solution))
    
    return results
# EVOLVE-BLOCK-END
