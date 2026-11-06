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

    def solve_transversal(n, edges, cliques):
        # Filter cliques of size >= 2
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # If MILP is available, use it for exact solution
        try:
            # Variables: x_v for v in 0..n-1
            num_vars = n
            # Objective: minimize sum(x_v)
            c = np.ones(num_vars)
            
            # Constraints: sum(x_v for v in clique) >= 1
            # A @ x >= 1
            A = np.zeros((len(relevant_cliques), num_vars))
            for i, clique in enumerate(relevant_cliques):
                for v in clique:
                    A[i, v] = 1
            
            # LinearConstraint(A, 1, np.inf)
            constraint = LinearConstraint(A, 1, np.inf)
            
            # Bounds: 0 <= x_v <= 1
            bounds = Bounds(lb=0, ub=1)
            
            # Integrality: all integer
            integrality = np.ones(num_vars)
            
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                # res.x contains the solution
                selected = [i for i in range(n) if res.x[i] > 0.5]
                return selected
            else:
                # Fallback to greedy if MILP fails
                pass
                
        except Exception:
            pass
            
        # Fallback: Greedy Set Cover
        # Map clique index to list of vertices
        # We want to pick vertices to cover all cliques.
        # Greedy: pick vertex that covers most uncovered cliques.
        
        # Precompute for each vertex, the list of clique indices it belongs to
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(relevant_cliques):
            for v in clique:
                vertex_to_cliques[v].append(i)
                
        covered = [False] * len(relevant_cliques)
        selected = []
        remaining_cliques = set(range(len(relevant_cliques)))
        
        while remaining_cliques:
            best_v = None
            best_count = -1
            for v in range(n):
                # Count how many remaining cliques v covers
                count = 0
                for ci in vertex_to_cliques[v]:
                    if ci in remaining_cliques:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
                    
            if best_count == 0:
                # Should not happen if input is valid, but break to avoid infinite loop
                break
                
            selected.append(best_v)
            # Remove covered cliques
            for ci in vertex_to_cliques[best_v]:
                if ci in remaining_cliques:
                    remaining_cliques.remove(ci)
                    
        return selected

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        cliques = bron_kerbosch(edges, n)
        transversal = solve_transversal(n, edges, cliques)
        out.append(sorted(transversal))
        
    return out
