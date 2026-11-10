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
                cliques.append(set(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            candidates = P - adj[u]
        elif X:
            u = max(X, key=lambda v: len(adj[v] & X))
            candidates = P  # P is empty, so this loop won't run
            # Actually if P is empty, we already returned above
            return
        else:
            return
        
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
    return cliques

def _solve_set_cover_greedy(vertices, cliques, n):
    """
    Solve the set cover problem: select minimum vertices to cover all cliques.
    Uses a weighted greedy approach with local search refinement.
    
    vertices: list of vertex indices
    cliques: list of sets of vertex indices (each clique must be hit)
    n: number of vertices
    """
    if not cliques:
        return set()
    
    num_cliques = len(cliques)
    
    # Map each vertex to the cliques it covers
    vertex_to_cliques = defaultdict(list)
    for ci, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].append(ci)
    
    # Precompute coverage as lists
    coverage = {v: vertex_to_cliques.get(v, []) for v in range(n)}
    
    def greedy_set_cover(coverage, num_cliques, n, rng):
        """Greedy set cover with optional randomization for tie-breaking."""
        selected = set()
        uncovered = set(range(num_cliques))
        
        # Track how many times each clique is covered by selected vertices
        # We just need to ensure all are covered
        
        while uncovered:
            best_v = None
            best_count = -1
            best_ties = []
            
            for v in range(n):
                if v in selected:
                    continue
                count = 0
                for ci in coverage[v]:
                    if ci in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
                    best_ties = [v]
                elif count == best_count and count > 0:
                    best_ties.append(v)
            
            if best_v is None or best_count == 0:
                # No progress possible - shouldn't happen if feasible
                break
            
            # Randomly pick from ties for diversity
            if len(best_ties) > 1:
                best_v = rng.choice(best_ties)
            
            selected.add(best_v)
            for ci in coverage[best_v]:
                if ci in uncovered:
                    uncovered.remove(ci)
        
        return selected
    
    def local_search(selected, coverage, cliques, num_cliques, n, max_rounds=5):
        """Try to remove vertices or swap to reduce solution size."""
        selected = set(selected)
        
        for _ in range(max_rounds):
            improved = False
            
            # Try removing each vertex
            for v in list(selected):
                # Check if all cliques still covered without v
                # Build coverage count for each clique
                cover_count = [0] * num_cliques
                for sv in selected:
                    if sv == v:
                        continue
                    for ci in coverage[sv]:
                        cover_count[ci] += 1
                
                # Check if any clique has cover_count 0
                all_covered = True
                for ci in range(num_cliques):
                    if cover_count[ci] == 0:
                        all_covered = False
                        break
                
                if all_covered:
                    selected.remove(v)
                    improved = True
                    break
            
            if not improved:
                break
        
        return selected
    
    # Run multiple greedy passes with different random seeds
    best_solution = None
    best_size = float('inf')
    
    for trial in range(10):
        rng = Random(42 + trial)
        sol = greedy_set_cover(coverage, num_cliques, n, rng)
        
        # Apply local search
        sol = local_search(sol, coverage, cliques, num_cliques, n)
        
        # Verify solution
        covered = set()
        for v in sol:
            for ci in coverage[v]:
                covered.add(ci)
        
        if covered == set(range(num_cliques)):
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
    
    if best_solution is None:
        # Fallback: pure greedy without local search
        rng = Random(42)
        best_solution = greedy_set_cover(coverage, num_cliques, n, rng)
    
    return best_solution

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
            selected = _solve_set_cover_greedy(range(n), cliques, n)
            out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END
