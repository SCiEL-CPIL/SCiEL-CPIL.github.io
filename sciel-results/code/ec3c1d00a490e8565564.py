import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(graph, n):
        """
        Finds all maximal cliques in a graph.
        graph: list of edges
        n: number of vertices
        Returns a list of sets, each set is a maximal clique.
        """
        # Build adjacency list and sets for fast intersection
        adj = [set() for _ in range(n)]
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if R: # Only add if not empty, though R can be empty for isolated vertices (size 1)
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot = None
            max_degree = -1
            for v in P.union(X):
                deg = len(P.intersection(adj[v]))
                if deg > max_degree:
                    max_degree = deg
                    pivot = v
            
            # Iterate over vertices in P \ N(pivot)
            for v in list(P - adj[pivot]):
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                
                bk(R_new, P_new, X_new)
                
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return [list(c) for c in cliques]

    def solve_transversal_bb(n, edges, cliques):
        """Branch and bound solver for minimum clique transversal."""
        # Filter cliques of size >= 2
        relevant_cliques = [set(c) for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # Precompute: for each vertex, which cliques it belongs to
        vertex_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(relevant_cliques):
            for v in clique:
                vertex_cliques[v].append(i)
        
        num_cliques = len(relevant_cliques)
        
        # Get initial upper bound using greedy
        def greedy_solution():
            covered = [False] * num_cliques
            selected = []
            remaining = set(range(num_cliques))
            
            while remaining:
                best_v = None
                best_count = -1
                for v in range(n):
                    count = sum(1 for ci in vertex_cliques[v] if ci in remaining)
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_count == 0:
                    break
                selected.append(best_v)
                for ci in vertex_cliques[best_v]:
                    if ci in remaining:
                        remaining.remove(ci)
            return len(selected), selected
        
        ub_size, ub_solution = greedy_solution()
        
        # Lower bound: maximum set of pairwise disjoint cliques
        # Each disjoint clique needs at least one vertex
        def compute_lower_bound(uncovered_cliques):
            """Compute lower bound using greedy maximal disjoint clique packing."""
            if not uncovered_cliques:
                return 0
            
            # Greedy: pick cliques that are disjoint from each other
            # Sort by size (smaller first for better packing)
            cliques_list = sorted(uncovered_cliques, key=len)
            chosen = []
            chosen_vertices = set()
            
            for clique in cliques_list:
                if not clique & chosen_vertices:
                    chosen.append(clique)
                    chosen_vertices |= clique
            
            return len(chosen)
        
        # Branch and bound
        best_size = ub_size
        best_solution = ub_solution.copy()
        
        # State: (selected_vertices, uncovered_clique_indices)
        # Use recursion with pruning
        
        def branch_and_bound(selected, uncovered):
            nonlocal best_size, best_solution
            
            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = selected.copy()
                return
            
            # Lower bound check
            lb = len(selected) + compute_lower_bound([relevant_cliques[i] for i in uncovered])
            if lb >= best_size:
                return
            
            # Choose a clique to branch on - pick smallest uncovered clique
            min_clique_idx = min(uncovered, key=lambda i: len(relevant_cliques[i]))
            clique = relevant_cliques[min_clique_idx]
            
            # Branch: try each vertex in this clique
            # Order vertices by how many other uncovered cliques they cover (most first)
            def vertex_priority(v):
                return sum(1 for ci in vertex_cliques[v] if ci in uncovered)
            
            for v in sorted(clique, key=vertex_priority, reverse=True):
                new_selected = selected + [v]
                # Remove all cliques that v covers
                new_uncovered = set(uncovered)
                for ci in vertex_cliques[v]:
                    new_uncovered.discard(ci)
                
                branch_and_bound(new_selected, new_uncovered)
                
                # If we already found a solution of size len(selected), 
                # no need to try more vertices from this clique
                if len(selected) >= best_size:
                    break
        
        branch_and_bound([], set(range(num_cliques)))
        
        return sorted(best_solution)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        transversal = solve_transversal_bb(n, edges, cliques)
        out.append(sorted(transversal))
        
    return out
